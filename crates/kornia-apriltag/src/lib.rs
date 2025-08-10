#![deny(missing_docs)]
//! # Kornia AprilTag

use std::collections::HashMap;

use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};
use kornia_imgproc::resize::resize_fast_mono;

use crate::{
    decoder::{decode_tags, Detection, GrayModelPair},
    errors::AprilTagError,
    quad::{fit_quads, FitQuadConfig},
    segmentation::{find_connected_components, find_gradient_clusters, GradientInfo},
    threshold::{adaptive_threshold, TileMinMax},
    union_find::UnionFind,
    utils::Pixel,
};

/// Error types for AprilTag detection.
pub mod errors;

/// Utility functions for AprilTag detection.
pub mod utils;

/// Thresholding utilities for AprilTag detection.
pub mod threshold;

/// image iteration utilities module.
pub(crate) mod iter;

/// Segmentation utilities for AprilTag detection.
pub mod segmentation;

/// Union-find utilities for AprilTag detection.
pub mod union_find;

/// AprilTag family definitions and utilities.
pub mod family;

/// Quad detection utilities for AprilTag detection.
pub mod quad;

/// Decoding utilities for AprilTag detection.
pub mod decoder;

// Re-export commonly used types
pub use family::{TagFamily, TagFamilyKind};

/// Configuration for decoding AprilTags.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTagsConfig {
    /// List of tag families to detect.
    pub tag_families: Vec<crate::TagFamily>,
    /// Configuration for quad fitting.
    pub fit_quad_config: FitQuadConfig,
    /// Whether to enable edge refinement before decoding.
    pub refine_edges_enabled: bool,
    /// Sharpening factor applied during decoding.
    pub decode_sharpening: f32,
    /// Whether normal border tags are present.
    pub normal_border: bool,
    /// Whether reversed border tags are present.
    pub reversed_border: bool,
    /// Minimum tag width at border among all families.
    pub min_tag_width: usize,
    /// Minimum difference between white and black pixels for thresholding.
    pub min_white_black_difference: u8,
    /// Downscale factor for input images.
    pub downscale_factor: usize,
}

const DEFAULT_DOWNSCALE_FACTOR: usize = 2;

impl DecodeTagsConfig {
    /// Creates a new `DecodeTagsConfig` with the given tag family kinds.
    pub fn new(tag_family_kinds: Vec<crate::TagFamilyKind>) -> Self {
        let mut tag_families = Vec::with_capacity(tag_family_kinds.len());
        let mut normal_border = false;
        let mut reversed_border = false;
        let mut min_tag_width = usize::MAX;

        tag_family_kinds.iter().for_each(|family_kind| {
            let family: crate::TagFamily = family_kind.into();
            if family.width_at_border < min_tag_width {
                min_tag_width = family.width_at_border;
            }
            normal_border |= !family.reversed_border;
            reversed_border |= family.reversed_border;

            tag_families.push(family);
        });

        if min_tag_width < 3 {
            min_tag_width = 3;
        }

        Self {
            tag_families,
            fit_quad_config: Default::default(),
            normal_border,
            refine_edges_enabled: true,
            decode_sharpening: 0.25,
            reversed_border,
            min_tag_width,
            min_white_black_difference: 5,
            downscale_factor: DEFAULT_DOWNSCALE_FACTOR,
        }
    }

    /// Creates a new `DecodeTagsConfig` with the given tag family kinds and hamming distance.
    ///
    /// # Arguments
    ///
    /// * `tag_family_kinds` - The tag families to use for detection
    /// * `hamming_distance` - Maximum hamming distance for error correction (0-3 recommended)
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_apriltag::{DecodeTagsConfig, TagFamilyKind};
    ///
    /// // Create config with Tag36H11 using hamming distance of 1 for faster processing
    /// let config = DecodeTagsConfig::with_hamming_distance(
    ///     vec![TagFamilyKind::Tag36H11],
    ///     1
    /// );
    /// ```
    pub fn with_hamming_distance(
        tag_family_kinds: Vec<crate::TagFamilyKind>,
        hamming_distance: u8,
    ) -> Self {
        let mut tag_families = Vec::with_capacity(tag_family_kinds.len());
        let mut normal_border = false;
        let mut reversed_border = false;
        let mut min_tag_width = usize::MAX;

        tag_family_kinds.iter().for_each(|family_kind| {
            let base_family: crate::TagFamily = family_kind.into();
            let family = crate::TagFamily::with_hamming_distance(base_family, hamming_distance);

            if family.width_at_border < min_tag_width {
                min_tag_width = family.width_at_border;
            }
            normal_border |= !family.reversed_border;
            reversed_border |= family.reversed_border;

            tag_families.push(family);
        });

        min_tag_width /= DEFAULT_DOWNSCALE_FACTOR;

        if min_tag_width < 3 {
            min_tag_width = 3;
        }

        Self {
            tag_families,
            fit_quad_config: Default::default(),
            normal_border,
            refine_edges_enabled: true,
            decode_sharpening: 0.25,
            reversed_border,
            min_tag_width,
            min_white_black_difference: 5,
            downscale_factor: DEFAULT_DOWNSCALE_FACTOR,
        }
    }

    /// Creates a `DecodeTagsConfig` with all supported tag families.
    pub fn all() -> Self {
        Self::new(crate::TagFamilyKind::all())
    }

    /// Creates a `DecodeTagsConfig` with all supported tag families and custom hamming distance.
    ///
    /// # Arguments
    ///
    /// * `hamming_distance` - Maximum hamming distance for error correction (0-3 recommended)
    pub fn all_with_hamming_distance(hamming_distance: u8) -> Self {
        Self::with_hamming_distance(crate::TagFamilyKind::all(), hamming_distance)
    }

    /// Adds a tag family to the configuration.
    pub fn add(&mut self, family: crate::TagFamily) {
        if family.width_at_border < self.min_tag_width {
            self.min_tag_width = family.width_at_border;
        }
        self.normal_border |= !family.reversed_border;
        self.reversed_border |= family.reversed_border;

        self.tag_families.push(family);
    }
}

/// Decoder for AprilTag detection and decoding.
pub struct AprilTagDecoder {
    config: DecodeTagsConfig,
    downscale_img: Option<Image<u8, 1, CpuAllocator>>,
    bin_img: Image<Pixel, 1, CpuAllocator>,
    tile_min_max: TileMinMax,
    uf: UnionFind,
    clusters: HashMap<(usize, usize), Vec<GradientInfo>>,
    gray_model_pair: GrayModelPair,
}

impl AprilTagDecoder {
    /// Returns a reference to the decoder configuration.
    #[inline]
    pub fn config(&self) -> &DecodeTagsConfig {
        &self.config
    }

    /// Adds a tag family to the decoder configuration.
    #[inline]
    pub fn add(&mut self, family: crate::TagFamily) {
        self.config.add(family);
    }

    /// Creates a new `AprilTagDecoder` with the given configuration and image size.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for decoding AprilTags.
    /// * `img_size` - The size of the image to be processed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `AprilTagDecoder` or an `AprilTagError`.
    pub fn new(config: DecodeTagsConfig, img_size: ImageSize) -> Result<Self, AprilTagError> {
        let (img_size, downscale_img) = if config.downscale_factor <= 1 {
            (img_size, None)
        } else {
            let new_size = ImageSize {
                width: img_size.width / config.downscale_factor,
                height: img_size.height / config.downscale_factor,
            };

            (
                new_size,
                Some(Image::from_size_val(new_size, 0, CpuAllocator)?),
            )
        };

        let bin_img = Image::from_size_val(img_size, Pixel::Skip, CpuAllocator)?;
        let tile_min_max = TileMinMax::new(img_size, 4);
        let uf = UnionFind::new(img_size.width * img_size.height);

        Ok(Self {
            config,
            downscale_img,
            bin_img,
            tile_min_max,
            uf,
            clusters: HashMap::new(),
            gray_model_pair: GrayModelPair::new(),
        })
    }

    /// Creates a new `AprilTagDecoder` with a custom hamming distance for error correction.
    ///
    /// # Arguments
    ///
    /// * `tag_family_kinds` - The tag families to use for detection
    /// * `hamming_distance` - Maximum hamming distance for error correction (0-3 recommended)
    /// * `img_size` - The size of the image to be processed
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `AprilTagDecoder` or an `AprilTagError`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kornia_apriltag::{AprilTagDecoder, TagFamilyKind};
    /// use kornia_image::ImageSize;
    ///
    /// // Create decoder with hamming distance of 1 for faster processing
    /// let decoder = AprilTagDecoder::with_hamming_distance(
    ///     vec![TagFamilyKind::Tag36H11],
    ///     1,
    ///     ImageSize { width: 640, height: 480 }
    /// ).unwrap();
    /// ```
    pub fn with_hamming_distance(
        tag_family_kinds: Vec<crate::TagFamilyKind>,
        hamming_distance: u8,
        img_size: ImageSize,
    ) -> Result<Self, AprilTagError> {
        let config = DecodeTagsConfig::with_hamming_distance(tag_family_kinds, hamming_distance);
        Self::new(config, img_size)
    }

    /// Decodes AprilTags from the provided grayscale image.
    ///
    /// # Arguments
    ///
    /// * `src` - The source grayscale image to decode tags from.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of `Detection` or an `AprilTagError`.
    ///
    /// # Note
    ///
    /// If you are running this method multiple times on the same decoder instance,
    /// you should call [`AprilTagDecoder::clear`] between runs to reset internal state.
    pub fn decode<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
    ) -> Result<Vec<Detection>, AprilTagError> {
        if let Some(downscale_img) = self.downscale_img.as_mut() {
            resize_fast_mono(
                src,
                downscale_img,
                kornia_imgproc::interpolation::InterpolationMode::Nearest,
            )?;

            // Step 1: Adaptive Threshold
            adaptive_threshold(
                downscale_img,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
            )?;
        } else {
            // Step 1: Adaptive Threshold
            adaptive_threshold(
                src,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
            )?;
        }

        // Step 2(a): Find Connected Components
        find_connected_components(&self.bin_img, &mut self.uf)?;

        // Step 2(b): Find Clusters
        find_gradient_clusters(&self.bin_img, &mut self.uf, &mut self.clusters);

        // Step 3: Quad Fitting
        let mut quads = fit_quads(&self.bin_img, &mut self.clusters, &self.config);

        // Step 4: Tag Decoding
        Ok(decode_tags(
            src,
            &mut quads,
            &mut self.config,
            &mut self.gray_model_pair,
        ))
    }

    /// Clears the internal state of the decoder for reuse.
    pub fn clear(&mut self) {
        self.uf.reset();
        self.clusters.clear();
        self.gray_model_pair.reset();
    }

    /// Returns a slice of tag families configured for detection.
    pub fn tag_families(&self) -> &[crate::TagFamily] {
        &self.config.tag_families
    }
}

/// Running the test on aarch64 crashes the CI
#[cfg(all(test, not(target_arch = "aarch64")))]
mod tests {
    use kornia_io::png::read_image_png_mono8;

    use crate::{family::TagFamilyKind, utils::Point2d, AprilTagDecoder, DecodeTagsConfig};

    fn test_tags(
        decoder: &mut AprilTagDecoder,
        expected_tag: TagFamilyKind,
        expected_quads: [Point2d<f32>; 4],
        images_dir: &str,
        file_name_starts_with: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tag_images = std::fs::read_dir(images_dir)?;

        for img in tag_images {
            let img = img?;
            let file_name = img.file_name();
            let file_name = file_name
                .to_str()
                .ok_or("Failed to convert file name to str")?;

            if file_name.starts_with(file_name_starts_with) {
                let file_path = img.path();

                let expected_id = file_name.strip_prefix(file_name_starts_with).unwrap();
                let expected_id = expected_id.strip_suffix(".png").unwrap();
                let Ok(expected_id) = expected_id.parse::<u16>() else {
                    // Currently we only support decoding id upto 65535 (u16::MAX) while some tag families
                    // like `TagCircle49H12` can support more than that.
                    continue;
                };

                if expected_id == u16::MAX {
                    continue;
                }

                let original_img = read_image_png_mono8(file_path)?;
                let detection = decoder.decode(&original_img)?;

                assert_eq!(detection.len(), 1, "Tag: {file_name}");
                let detection = &detection[0];

                assert_eq!(detection.id, expected_id);
                assert_eq!(detection.tag_family_kind, expected_tag);

                for (point, expected) in detection.quad.corners.iter().zip(expected_quads.iter()) {
                    assert!(
                        (point.y - expected.y).abs() <= 0.1,
                        "Tag: {}, Got y: {}, Expected: {}",
                        file_name,
                        point.y,
                        expected.y
                    );
                    assert!(
                        (point.x - expected.x).abs() <= 0.1,
                        "Tag: {}, Got x: {}, Expected: {}",
                        file_name,
                        point.x,
                        expected.x
                    );
                }

                decoder.clear();
            }
        }

        Ok(())
    }

    #[test]
    fn test_tag16_h5() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag16H5]);
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 10.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 10.0, y: 40.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::Tag16H5,
            expected_quad,
            "../../tests/data/apriltag-imgs/tag16h5/",
            "tag16_05_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tag25_h9() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag25H9]);
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 10.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 10.0, y: 45.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::Tag25H9,
            expected_quad,
            "../../tests/data/apriltag-imgs/tag25h9/",
            "tag25_09_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tag36_h11() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]);
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Point2d { x: 50.0, y: 10.0 },
            Point2d { x: 50.0, y: 50.0 },
            Point2d { x: 10.0, y: 50.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::Tag36H11,
            expected_quad,
            "../../tests/data/apriltag-imgs/tag36h11/",
            "tag36_11_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagcircle21h7() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCircle21H7]);
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 15.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 15.0, y: 40.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagCircle21H7,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagCircle21h7/",
            "tag21_07_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagcircle49h12() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCircle49H12]);
        let mut decoder = AprilTagDecoder::new(config, [65, 65].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 20.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 20.0, y: 45.0 },
            Point2d { x: 20.0, y: 20.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagCircle49H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagCircle49h12/",
            "tag49_12_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagcustom48_h12() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCustom48H12]);
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 15.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 15.0, y: 45.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagCustom48H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagCustom48h12/",
            "tag48_12_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagstandard41_h12() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagStandard41H12]);
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 15.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 15.0, y: 40.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagStandard41H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagStandard41h12/",
            "tag41_12_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagstandard52_h13() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagStandard52H13]);
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 15.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 15.0, y: 45.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagStandard52H13,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagStandard52h13/",
            "tag52_13_",
        )?;

        Ok(())
    }
}
