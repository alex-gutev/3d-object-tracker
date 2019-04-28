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

#include "yuv_reader.h"

yuv_reader::yuv_reader(std::string file, size_t w, size_t h) : yuv_reader(w, h) {
    stream.open(file, std::ios::in | std::ios::binary);
}

yuv_reader::yuv_reader(size_type w, size_type h)
        : width(w), height(h),
          ycrcb(h, w, CV_8UC3), y(h, w, CV_8UC1), cb(h, w, CV_8UC1), cr(h, w, CV_8UC1),
          cb_half(h/2, w/2, CV_8UC1), cr_half(h/2, w/2, CV_8UC1) {
    stream.exceptions(std::ios::failbit);
}

yuv_reader::yuv_reader() {
    stream.exceptions(std::ios::failbit);
}


void yuv_reader::dims(size_type w, size_type h) {
    width = w;
    height = h;

    create_buffers();
}

void yuv_reader::open(std::string file) {
    stream.open(file, std::ios::in | std::ios::binary);
}

bool yuv_reader::read(cv::Mat &bgr) {
    size_t npixels = width * height;

    if (stream.eof()) return false;

    try {
        stream.read((char *)y.data, npixels);
        stream.read((char *)cb_half.data, npixels/4);
        stream.read((char *)cr_half.data, npixels/4);
    }
    catch (std::ios_base::failure &e) {
        if (stream.eof()) {
            return false;
        }

        throw e;
    }

    // Upscale blue and red images
    cv::resize(cb_half, cb, cb.size(), 4, 2, CV_INTER_NN);
    cv::resize(cr_half, cr, cr.size(), 4, 2, CV_INTER_NN);

    // Merge YUV components into one 3-channel image
    std::vector<cv::Mat> mats({y, cr, cb});
    cv::merge(mats, ycrcb);

    // Convert YUV to RGB
    cv::cvtColor(ycrcb, bgr, cv::COLOR_YCrCb2BGR);

    return true;
}

void yuv_reader::create_buffers() {
    ycrcb.create(height, width, CV_8UC3);

    y.create(height, width, CV_8UC1);
    cb.create(height, width, CV_8UC1);
    cr.create(height, width, CV_8UC1);

    cb_half.create(height/2, width/2, CV_8UC1);
    cr_half.create(height/2, width/2, CV_8UC1);
}
