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

#include "view.h"

view::view(cv::Matx<float, 4, 4> matrix_intrinsic, cv::Matx<float, 4, 4> matrix_extrinsic)
        : mat_intrinsic(matrix_intrinsic), mat_extrinsic(matrix_extrinsic),
          intrinsic_inverse(matrix_intrinsic.inv()), extrinsic_inverse(matrix_extrinsic.inv()) {}

void view::open(const std::string &color, const std::string &depth) {
    color_video.open(color);
    depth_video.open(depth);

    auto dims = color_video.dims();

    color_frame.create(dims.second, dims.first, CV_8UC3);
    depth_frame.create(dims.second, dims.first, CV_8UC3);
}

bool view::read() {
    if (!color_video.read(color_frame))
        return false;

    if (!depth_video.read(depth_frame))
        return false;

    // Convert depth image from 3-channel image to grey-scale
    cv::cvtColor(depth_frame, depth_frame, cv::COLOR_BGR2GRAY);

    return true;
}


cv::Vec4f view::pixel_to_camera(float x, float y, float z) {
    // Reverse Perspective Projection
    cv::Vec4f p(x * z, y * z, z, 1);

    // Transform to camera space
    return inv_intrinsic_matrix() * p;
}

cv::Vec4f view::pixel_to_world(float x, float y, float z) {
    // Transform to camera space

    return inv_extrinsic_matrix() * pixel_to_camera(x, y, z);
}

cv::Vec4f view::camera_to_pixel(cv::Vec4f p) {
    cv::Vec4f pt = intrinsic_matrix() * p;

    pt[0] /= pt[2];
    pt[1] /= pt[2];

    return pt;
}

cv::Vec4f view::world_to_pixel(cv::Vec4f p) {
    return camera_to_pixel(extrinsic_matrix() * p);
}


cv::Point view::transform(cv::Point coord, const view &next_view) {
    // Get depth map intensity value of pixel
    uchar disparity = depth_frame.at<uchar>(coord);

    // Map pixel to next view
    cv::Point3f pt = transform(coord, disparity_to_depth(disparity), next_view);

    // Return 2D coordinate
    return cv::Point(pt.x, pt.y);
}

cv::Point3f view::transform(cv::Point px, float depth, const view &next_view) {
    // Create 3D homogeneous coordinates, reversing perspective
    // projection with the origin at the bottom-left of the frame
    // image
    cv::Vec<float, 4> pt(px.x * depth, (color_frame.rows - px.y) * depth, depth, 1);

    // Transform point to corresponding point in next view
    pt = next_view.mat_intrinsic * next_view.mat_extrinsic * extrinsic_inverse * intrinsic_inverse * pt;

    return cv::Point3f(pt[0]/pt[2], color_frame.rows - pt[1]/pt[2], pt[2]);
}


float view::disparity_to_depth(uchar disparity) {
    if (min_z != 0 & max_z != 0)
        return 1.0f / ((disparity/255.0f)*(1.0f/min_z - 1.0f/max_z) + 1.0f/max_z);
    else
        return disparity + 1;
}

cv::Mat view::disparity_to_depth(cv::Mat img) {
    if (min_z != 0 & max_z != 0) {
        cv::Mat m;

        img.convertTo(m, CV_32FC1);

        return 1.0f / ((m/255.0f) * (1.0f/min_z - 1.0f/max_z) + 1.0f/max_z);
    }
    else {
        return img + 1;
    }
}


const cv::Matx<float, 4, 4> &view::intrinsic_matrix() const {
    return mat_intrinsic;
}

void view::intrinsic_matrix(const cv::Matx<float, 4, 4> &intrinsic_matrix) {
    mat_intrinsic = intrinsic_matrix;
    intrinsic_inverse = intrinsic_matrix.inv();
}

const cv::Matx<float, 4, 4> &view::extrinsic_matrix() const {
    return mat_extrinsic;
}

void view::extrinsic_matrix(const cv::Matx<float, 4, 4> &world_matrix) {
    mat_extrinsic = world_matrix;
    extrinsic_inverse = mat_extrinsic.inv();
}

std::pair<int, int> view::dims() const {
    return color_video.dims();
}


void view::dims(int width, int height) {
    color_video.dims(width, height);
    depth_video.dims(width, height);
}
