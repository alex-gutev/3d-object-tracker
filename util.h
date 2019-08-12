/*
 * Copyright (C) 2019  Alexander Gutev <alex.gutev@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef AGTRACK_UTIL_H
#define AGTRACK_UTIL_H

/**
 * Utility functions used throughout the project.
 */

#include <algorithm>

/**
 * Clamps a value to the range [@a min, @a max]
 *
 * @param x The value to clamp.
 * @param min The lower-bound of the range.
 * @param max The upper-bound of the range.
 *
 * @return The clamped value
 */
template<typename T>
T clamp(T x, T min, T max) {
    return std::min(std::max(x, min), max);
}

/**
 * Clamps the region @a r to be within an image with origin at (0,0)
 * and size @a size.
 *
 * @param r Region to clamp.
 * @param size Size of the image.
 */
cv::Rect clamp_region(cv::Rect r, cv::Size size);

/**
 * Computes the linear binary pattern of @a img.
 *
 * @param img Image of which to compute the LBP.
 * @param ksize LBP kernel size.
 *
 * @return Matrix in which each element contains the LBP at the
 *   corresponding pixel within @a img.
 */
cv::Mat compute_lbp(cv::Mat img, int ksize);


/**
 * Segments the image @a color using the watershed algorithm with the
 * markers determined by the depth image @a depth.
 *
 * @param depth Single-channel depth image used to determine markers.
 *
 * @param color Three-channel image which is actually segmented.
 *
 * @param markers Output matrix containing segmented image.
 *
 * @return The number of objects.
 */
size_t watershed(cv::Mat depth, cv::Mat color, cv::Mat &markers);

/**
 * Returns the @a percent percentile of the values in the image,
 * within the mask.
 *
 * @param img       The image
 * @param percent   The percentile
 * @param mask      The mask of the pixels which are considered
 *
 * @return The percentile value
 */
double percentile(cv::Mat img, double percent, cv::Mat mask);


/**
 * Computes the magnitude of a vector.
 *
 * @param v The vector
 *
 * @return The magnitude
 */
template <typename T>
float magnitude(T v) {
    float mag = 0;

    for (int i = 0; i < v.cols; i++) {
        mag += v[i] * v[i];
    }

    return sqrtf(mag);
}

/**
 * Computes the magnitude of a 3D point vector.
 *
 * @param pt The vector
 * @return The magnitude
 */
float magnitude(cv::Point3f pt);


#endif /* AGTRACK_UTIL_H */

// Local Variables:
// mode: c++
// End:
