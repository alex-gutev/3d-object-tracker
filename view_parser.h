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

#ifndef FYP_VIEW_PARSER_H
#define FYP_VIEW_PARSER_H

#include <iostream>
#include <exception>
#include <tuple>

#include <opencv2/imgproc.hpp>

#include "view.h"

/**
 * Parses view parameters from a file.
 *
 * Parameters include:
 *
 *   The path to the colour and depth
 *   video files
 *
 *   The video dimensions
 *
 *   The z-near and z-far
 *
 *   The intrinsic and fundamental matrices
 */
class view_parser {
    /**
     * Input file stream
     */
    std::ifstream stream;

    /**
     * Last line read from the file.
     */
    std::string line;

    /**
     * Parses the video dimensions from the last read line (stored in
     * @a line).
     *
     * @return A pair <width, height>
     */
    std::pair<int, int> parse_dims();

    /**
     * Parse a pair of values.
     *
     * @return The pair of values.
     */
    template <typename T>
    std::pair<T,T> parse_pair();

    /**
     * Parse the z-near and z-far
     *
     * @return Returns a pair with <z-near, z-far>
     */
    std::pair<float, float> parse_z_bounds();

    /**
     * Reads a 4x4 matrix from the file.
     *
     * @return The matrix
     */
    cv::Matx<float, 4, 4> read_matrix();

    /**
     * Reads one matrix row (a 4 element vector) from the file.
     *
     * @return The matrix row.
     */
    cv::Vec<float, 4> read_row();

    /**
     * Sets a row of a 4x4 matrix to the values in a 4 element vector.
     *
     * @param mat   Reference to the matrix
     * @param row   The index of the row
     * @param vec   The 4 element vector containing the values
     */
    static void set_row(cv::Matx<float, 4, 4> &mat, int row, cv::Vec<float, 4> vec);

public:

    /**
     * Exception thrown when the file is missing parameters or the
     * parameters are not in the correct format.
     */
    class incorrect_format_error : std::exception {};

    /**
     * Creates a view parser for a given file.
     *
     * @param file  Path to the file.
     */
    view_parser(const std::string file) : stream(file) {}

    /**
     * Returns true if the file contains more view parameters to
     * parse, false if EOF is reached.
     *
     * @return true if EOF has not yet been reached.
     */
    bool next();

    /**
     * Parses a view from the file and sets the view object's
     * parameters.
     *
     * This method should only be called if true was returned from the
     * previous call to next()
     *
     * @param v Reference to the view object.
     */
    void parse(view &v);

private:

    /**
     * Signals a parameter format error by throwing an exception.
     */
    void error() {
        throw incorrect_format_error();
    }
};


// Template Implementation

template<typename T>
std::pair<T, T> view_parser::parse_pair() {
    T x, y;

    std::stringstream line_stream(line);

    if (!(line_stream >> x)) error();
    if (!(line_stream >> y)) error();

    return std::make_pair(x, y);
}


#endif //FYP_VIEW_PARSER_H
