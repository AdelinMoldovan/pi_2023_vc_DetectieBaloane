#include "stdafx.h"
#include "util.h"




std::tuple<cv::Mat, cv::Mat, cv::Mat> convertToHSV(const cv::Mat& inputImage) {
	int height = inputImage.rows;
	int width = inputImage.cols;
	cv::Mat_<uchar> h = cv::Mat_<uchar>(height, width);
	cv::Mat_<uchar> s = cv::Mat_<uchar>(height, width);
	cv::Mat_<uchar> v = cv::Mat_<uchar>(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cv::Vec3b pixel = inputImage.at<cv::Vec3b>(i, j);
			float r = pixel[2];
			float g = pixel[1];
			float b = pixel[0];
			r /= 255;
			g /= 255;
			b /= 255;
			float M = max(max(r, g), b);
			float m = min(min(r, g), b);
			float C = M - m;
			float V = M;
			float S;
			float H;
			if (V != 0)
				S = C / V;
			else
				S = 0;

			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else
				H = 0;
			if (H < 0)
				H = H + 360;

			h(i, j) = H * 255 / 360;
			s(i, j) = S * 255;
			v(i, j) = V * 255;
		}
	}

	return std::make_tuple(h, s, v);
}


inline bool operator<(const cv::Vec3b& lhs, const cv::Vec3b& rhs) {
	return lhs[0] < rhs[0] ||
		(lhs[0] == rhs[0] && lhs[1] < rhs[1]) ||
		(lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] < rhs[2]);
}




