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

#ifndef FYP_YUV_READ_H
#define FYP_YUV_READ_H

#include <fstream>
#include <tuple>

#include <cv.hpp>

/**
 * Reads a YUV video file encoded in I420 format.
 */
class yuv_reader {
public:
    typedef int size_type;

private:
    /**
     * Video file input stream
     */
    std::ifstream stream;

    /**
     * Frame image width
     */
    size_type width;
    /**
     * Frame image height
     */
    size_type height;

    /**
     * Luminance (Y) image
     */
    cv::Mat y;
    /**
     * Upscaled blue (U) image
     */
    cv::Mat cb;
    /**
     * Upscaled red (V) image
     */
    cv::Mat cr;

    /**
     * Blue (U) image
     *
     * Half the size of the Y image.
     */
    cv::Mat cb_half;
    /**
     * Red (V) image
     *
     * Half the size of the Y image.
     */
    cv::Mat cr_half;

    /**
     * Full YUV image.
     */
    cv::Mat ycrcb;


    /**
     * Creates the image buffers
     */
    void create_buffers();

public:

    /**
     * Constructors
     */
    yuv_reader();
    yuv_reader(size_type w, size_type h);
    yuv_reader(std::string file, size_t w, size_t h);


    /**
     * @return The dimensions of the video frame images
     */
    std::pair<yuv_reader::size_type, yuv_reader::size_type> dims() const {
        return std::make_pair(width, height);
    };

    /**
     * Sets the video frame image dimensions
     *
     * @param w Width
     * @param h Height
     */
    void dims(size_type w, size_type h);

    /**
     * Opens the video file
     *
     * @param file Path to the video file
     */
    void open(std::string file);

    /**
     * Reads the next frame image from the video file.
     *
     * @param mat Matrix into which the frame image converted to RGB
     *            is stored

     * @return Returns true if the frame was read successfully, false
     *         if EOF is reached.
     */
    bool read(cv::Mat &mat);

    /**
     * Closes the file input stream.
     */
    void close() {
        stream.close();
    }
};


#endif //FYP_YUV_READ_H
