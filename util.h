#pragma once
#ifndef UTIL_H
#define UTIL_H
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>

using namespace cv;

std::tuple<cv::Mat, cv::Mat, cv::Mat> convertToHSV(const cv::Mat& inputImage);
inline bool inside(Mat img, int r, int c) {
	return (r < img.rows) && (r >= 0) && (c >= 0) && (c < img.cols);
}

inline bool operator<(const cv::Vec3b& lhs, const cv::Vec3b& rhs);

struct Vec3bComparator {
	bool operator()(const Vec3b lhs, const Vec3b rhs) const {
		return lhs < rhs;
	}
};
Mat_<Vec3b> colorCodeLabels(Mat_<int> labels);

template<typename T, typename V>
Mat_<V> map_pixels(const Mat_<T> img, std::function<V(T)> f) {
	Mat_<V> result(img.rows, img.cols);
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			result(i, j) = f(img(i, j));
		}
	}
	return result;
}
#endif