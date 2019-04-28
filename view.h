/*
 * Copyright (C) 2018  Alexander Gutev <alex.gutev@gmail.com>
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

#ifndef FYP_VIEW_H
#define FYP_VIEW_H

#include <opencv2/imgproc.hpp>

#include "yuv_reader.h"

/**
 * Stores the view's colour and depth frame images, the camera
 * parameters and provides functionality for mapping pixels between
 * views.
 */
class view {
    /**
     * Z-Near plane
     */
    float min_z = 42;
    /**
     * Z-Far plane
     */
    float max_z = 147;

    /**
     * Color video reader.
     */
    yuv_reader color_video;
    /**
     * Depth video reader.
     */
    yuv_reader depth_video;

    /**
     * The current frame of the color video.
     */
    cv::Mat color_frame;
    /**
     * The current frame of the depth video.
     */
    cv::Mat depth_frame;


    /* Camera Parameters */

    /**
     * Camera Intrinsic Matrix
     *
     * Transforms from camera space coordinates to pixel coordinates.
     */
    cv::Matx<float, 4, 4> mat_intrinsic;
    /**
     * Intrinsic matrix inverse
     */
    cv::Matx<float, 4, 4> intrinsic_inverse;

    /**
     * Camera rotation and translation Matrix (4x4)
     *
     * Transforms from world space to camera space.
     */
    cv::Matx<float, 4, 4> mat_extrinsic;
    /**
     * Extrinsic matrix inverse.
     */
    cv::Matx<float, 4, 4> extrinsic_inverse;

public:

    view() {}

    /**
     * Constructs a view with given camera matrix parameters.
     *
     * @param matrix_intrinsic Camera intrinsic Matrix
     * @param matrix_extrinsic Camera extrinsic Matrix
     */
    view(cv::Matx<float, 4, 4> matrix_intrinsic, cv::Matx<float, 4, 4> matrix_extrinsic);

    /**
     * @return The Z-near plane
     */
    float z_near() const {
        return min_z;
    }
    /**
     * Sets the value of the Z-near plane
     *
     * @param z The Z-near coordinate
     */
    void z_near(float z) {
        min_z = z;
    }

    /**
     * @return The Z-far plane
     */
    float z_far() const {
        return max_z;
    }
    /**
     * Sets the value of the Z-far plane
     *
     * @param z The Z-far coordinate
     */
    void z_far(float z) {
        max_z = z;
    }


    /**
     * @return The intrinsic matrix
     */
    const cv::Matx<float, 4, 4> &intrinsic_matrix() const;
    /**
     * @return The intrinsic matrix inverse
     */
    const cv::Matx<float, 4, 4> &inv_intrinsic_matrix() const {
        return intrinsic_inverse;
    }

    /**
     * Sets the intrinsic matrix.
     *
     * @param intrinsic_matrix  The matrix
     */
    void intrinsic_matrix(const cv::Matx<float, 4, 4> &intrinsic_matrix);


    /**
     * @return The extrinsic matrix
     */
    const cv::Matx<float, 4, 4> &extrinsic_matrix() const;
    /**
     * @return The extrinsic matrix inverse
     */
    const cv::Matx<float, 4, 4> &inv_extrinsic_matrix() const {
        return extrinsic_inverse;
    };

    /**
     * Sets the extrinsic matrix.
     *
     * @param world_matrix The matrix
     */
    void extrinsic_matrix(const cv::Matx<float, 4, 4> &world_matrix);


    /**
     * Returns the video frame dimensions.
     *
     * @return Dimensions as a pair <width, height>
     */
    std::pair<int,int> dims() const;

    /**
     * Sets the video frame dimensions.
     *
     * This method should be called before the video files are opened
     * (before open(...) is called).
     *
     * @param width     Video width
     * @param height    Video height
     */
    void dims(int width, int height);


    /**
     * Opens the video files.
     *
     * The dimensions of the videos should be set (using dims(...))
     * before calling this method.
     *
     * @param color Path to the colour video file
     * @param depth Path to the depth video file
     */
    void open(const std::string &color, const std::string &depth);

    /**
     * Reads a frame from the colour and depth video files.
     *
     * @return True if a frame was read from both video files
     *          successfully, false if EOF was reached.
     */
    bool read();

    /**
     * @return The current color video frame.
     */
    cv::Mat& color() {
        return color_frame;
    }
    /**
     * @return The current depth video frame.
     */
    cv::Mat& depth() {
        return depth_frame;
    }

    /**
     * Converts from a depth map intensity value to the actual
     * z-coordinate value of the pixel
     *
     * @param disparity The depth map intensity value
     * @return The value of the z-coordinate of the pixel
     */
    float disparity_to_depth(uchar disparity);

    /**
     * Transforms the coordinates, of a pixel in the view, to the
     * coordinates of the pixel's position in another view's image.
     *
     * @param coord     The coordinates of the pixel in the current view.
     * @param next_view The next view.
     *
     * @return The coordinates of the matching pixel in next_view.
     */
    cv::Point transform(cv::Point coord, const view &next_view);

    /**
     * Transforms the coordinates, of a pixel in the view, to the
     * coordinates of the pixel's position in another view's image.
     *
     * @param px         The x,y coordinates of the pixel
     * @param depth      The z-coordinate of the pixel
     * @param next_view  The next view
     *
     * @return The 3D coordinates of the pixel in the next view
     */
    cv::Point3f transform(cv::Point px, float depth, const view &next_view);
};


#endif //FYP_VIEW_H
