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

#include <sstream>

#include "view_parser.h"

bool view_parser::next() {
    while (std::getline(stream, line)) {
        if (!line.empty())
            return true;
    }

    return false;
}

void view_parser::parse(view &v) {
    // Parse video dimensions
    auto dims = parse_dims();
    v.dims(dims.first, dims.second);

    // Read colour video path
    if (!next()) error();
    std::string color_video = line;

    // Read depth video path
    if (!next()) error();

    // Open colour and depth video files
    v.open(color_video, line);

    // Parse z-near and z-far
    auto z_bounds = parse_z_bounds();

    v.z_near(z_bounds.first);
    v.z_far(z_bounds.second);

    // Parse intrinsic and extrinsic matrices
    v.intrinsic_matrix(read_matrix());
    v.extrinsic_matrix(read_matrix());
}

std::pair<int, int> view_parser::parse_dims() {
    return parse_pair<int>();
}

std::pair<float, float> view_parser::parse_z_bounds() {
    if (!next()) error();

    return parse_pair<float>();
}

cv::Matx<float, 4, 4> view_parser::read_matrix() {
    cv::Matx<float, 4, 4> mat;
    cv::Vec<float, 4> row;

    row = read_row();
    set_row(mat, 0, row);

    row = read_row();
    set_row(mat, 1, row);

    row = read_row();
    set_row(mat, 2, row);

    row = read_row();
    set_row(mat, 3, row);

    return mat;
}

void view_parser::set_row(cv::Matx<float, 4, 4> &mat, int row, cv::Vec<float, 4> values) {
    mat(row, 0) = values[0];
    mat(row, 1) = values[1];
    mat(row, 2) = values[2];
    mat(row, 3) = values[3];
}

cv::Vec<float, 4> view_parser::read_row() {
    if (!next())
        error();

    cv::Vec<float, 4> row;

    std::stringstream line_stream(line);

    if (!(line_stream >> row[0])) error();
    if (!(line_stream >> row[1])) error();
    if (!(line_stream >> row[2])) error();
    if (!(line_stream >> row[3])) error();

    return row;
}
