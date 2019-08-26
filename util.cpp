/*
 * Copyright (C) 2018-2019  Alexander Gutev <alex.gutev@gmail.com>
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

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <numeric>

#include "util.h"

static unsigned int mat_as_num(cv::Mat m) {
    unsigned int value = 0;

    auto size = m.size();

    for (size_t y = 0; y < size.height; ++y) {
        const unsigned char * row = m.ptr<unsigned char>(y);

        for (size_t x = 0; x < size.width; ++x) {
            value = value * 2 + row[x];
        }
    }

    return value;
}

cv::Rect clamp_region(cv::Rect r, cv::Size size) {
    r.x = clamp(r.x, 0, size.width - 1);
    r.y = clamp(r.y, 0, size.height - 1);

    r.width = clamp(r.width, 0, size.width - r.x);
    r.height = clamp(r.height, 0, size.height - r.y);

    return r;
}

cv::Rect clamp_rect(cv::Rect r, cv::Size size) {
    cv::Point tl(r.x, r.y);
    cv::Point br(r.x + r.width, r.y + r.height);

    tl.x = clamp(tl.x, 0, size.width - 1);
    tl.y = clamp(tl.y, 0, size.height - 1);

    br.x = clamp(br.x, 1, size.width);
    br.y = clamp(br.y, 1, size.height);

    return cv::Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
}

cv::Mat compute_lbp(cv::Mat img, int ksize) {
    auto size = img.size();
    cv::Mat out = cv::Mat::zeros(size, CV_32SC1);

    for (size_t y = 0; y < size.height - ksize; y += ksize) {
        for (size_t x = 0; x < size.width - ksize; x += ksize) {
            cv::Mat block = cv::Mat::zeros(cv::Size(ksize, ksize), CV_8UC1);

            int bx = std::max(ksize / 2 - (int)x, 0);
            int by = std::max(ksize / 2 - (int)y, 0);

            img(cv::Rect(x, y, ksize - bx, ksize - by)).copyTo(block(cv::Rect(bx, by, ksize - bx, ksize - by)));

            // std::cout << block << std::endl;

            auto threshold = block.at<unsigned char>(ksize / 2, ksize / 2);

            cv::threshold(block, block, threshold, 1, cv::THRESH_BINARY);

            auto val = mat_as_num(block);

            out.at<unsigned int>(y, x) = val;
        }
    }

    return out;
}

size_t watershed(cv::Mat depth, cv::Mat color, cv::Mat &markers) {
    // Threshold Depth Image

    cv::Mat bin;
    cv::threshold(depth, bin, 0, 255, cv::THRESH_OTSU);

    // Remove noise and small objects
    cv::Mat k1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k1);

    // Get border between objects and background
    cv::Mat border;
    cv::Mat objects;

    cv::dilate(bin, border, k1, cv::Point(-1,-1), 5);
    cv::erode(border, objects, k1);

    border -= objects;


    // Distance Transform
    cv::Mat dist;
    cv::distanceTransform(bin, dist, cv::DIST_L2, 3);

    cv::normalize(dist, dist, 0, 255, cv::NORM_MINMAX);
    dist.convertTo(dist, CV_8U);

    // Threshold to separate objects from background
    cv::threshold(dist, dist, 180, 255, cv::THRESH_BINARY);


    // Find blobs
    size_t n = cv::connectedComponents(dist, markers);

    // Add additional marker for border
    markers.setTo(n, border == 255);

    // Perform Watershedding

    markers.convertTo(markers, CV_32S);
    cv::watershed(color, markers);

    return n + 1;
}

double percentile(cv::Mat img, double percent, cv::Mat mask) {
    double min, max;

    cv::minMaxIdx(img, &min, &max);

    int channels[] = {0};
    int histSize[] = {256};
    float hranges[] = {(float)min, (float)max};
    const float *ranges[] = {hranges};

    cv::Mat hist;

    // Calculate histogram
    cv::calcHist(&img, 1, channels, mask, hist, 1, histSize, ranges);

    int total = 0;

    // Calculate the number of the pixel corresponding to the percentile
    double percentile = std::accumulate(hist.begin<float>(), hist.end<float>(), 0) * percent;

    int i = 0;
    for (auto it = hist.begin<float>(), end = hist.end<float>(); it != end; ++it) {
        int bin = *it;
        int prev = total;

        total += bin;

        // If the number of pixels exceeds the percentile exit loop
        if (total >= percentile) {
           float percent = (percentile - prev) / bin;

           return ((i + percent) / 255.0f) * (max - min) + min;
        }

        i++;
    }

    return -1;
}

float magnitude(cv::Point3f pt) {
    return sqrtf(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
}
