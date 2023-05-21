// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "common.h"
#include "test.h"
#include "util.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <random>
#include <functional>
#include <type_traits>

void testFindBalloons()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src;
		src = imread(fname, IMREAD_COLOR);
		const auto baloons = getBalloons(fname);
		drawBalloons(src, baloons);
		imshow("opened image", src);
		waitKey();
	}
}
template <typename ColorType>
class Blob {
public:
	int area;
	float perimeter;
	float circularity;
	ColorType color;
	Blob(int area, float perimeter, float circularity) :
		area(area), perimeter(perimeter), circularity(circularity) {};
	Blob(int area, float perimeter, float circularity, ColorType color) :
		area(area), perimeter(perimeter), circularity(circularity), color(color) {};
};

template <typename T>
inline Point getNextPoint(Mat_<T> img, Point p, int& dir);
template <typename T>
double perimeter(const Mat_<T>& img, T label);
template <typename T, typename Compare = std::less<T>>
Mat_<T> filter(Mat_<T> img, T background, std::function<bool(T)> predicate);
inline double circularity(int area, double perimeter) {
	return PI * 4 * area / (perimeter * perimeter);
}


template <typename T>
inline Point getNextPoint(Mat_<T> img, Point p, int& dir) {
	static Point neighbours[] = { { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, -1 }, { 1, -1 } };
	T label = img(p);
	dir = (dir & 1) ? (dir + 6) % 8 : (dir + 7) % 8;
	Point next = p + neighbours[dir];
	int i = 0;
	while (i < 8 && (inside(img, next.y, next.x) ? (img(next) != label) : true)) {
		dir++;
		dir %= 8;
		next = p + neighbours[dir];
		i++;
	}
	if (i > 7) return { -1, -1 };
	return next;
}
template <typename T>
double perimeter(const Mat_<T>& img, T label) {
	Point start;
	double contour = 0;
	const double SQRT2 = 1.41421356237;
	static const Point NOT_FOUND(-1, -1);
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img(i, j) == label) {
				start = { j, i };
				int dir = 7;
				Point second = getNextPoint(img, start, dir), previous = second, current = second;
				if (second == NOT_FOUND) return 1;
				do {
					contour += (dir & 1) ? SQRT2 : 1.0;
					previous = current;
					current = getNextPoint(img, current, dir);
				} while (previous != start || current != second);
				goto END;
			}
		}
	}
END:
	return contour;
}

template <typename T, typename Compare = std::less<T>>
Mat_<T> filter(Mat_<T> img, T background, std::function<bool(T)> predicate) {
	std::set<T, Compare> passed, failed;
	auto dst = img.clone();
	for (auto& label : dst) {
		if (label != background && passed.count(label) == 0) {
			if (failed.count(label) == 0) {
				if (predicate(label)) {
					passed.insert(label);
				}
				else {
					failed.insert(label);
					label = background;
				}
			}
			else {
				label = background;
			}
		}
	}
	return dst;
}
template <typename T>
Mat_<T> separate(const Mat_<T> img, const Point& seed, const T background) {
	static Point neighbours[] = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
	T label = img(seed);
	std::queue<Point> q;
	Mat_<T> object(img.rows, img.cols, background);;
	object(seed) = label;
	q.push(seed);
	while (!q.empty()) {
		Point p = q.front();
		q.pop();
		for (int k = 0; k < 8; ++k) {
			Point neighbour = p + neighbours[k];
			if (inside(img, neighbour.y, neighbour.x) && object(neighbour) == label) {
				object(neighbour.y, neighbour.x) = label;
				q.push(neighbour);
			}
		}
	}
	return object;
}

template <typename T, typename Compare = std::less<T>>
Mat_<T> map_objects(
	const Mat_<T>& img,
	T background,
	std::function<Mat_<T>(Mat_<T>)> map_object,
	std::function<void(Mat_<T>&, const Mat_<T>&, const T)> mix) {

	std::set<T, Compare> processed;
	Mat_<T> dst(img.rows, img.cols, background);

	int height = img.rows;
	int width = img.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (dst(i, j) != background && processed.count(dst(i, j)) == 0) {
				processed.insert(dst(i, j));
				auto object = separate(dst, { j, i }, background);
				auto newObject = map_object(object);
				mix(dst, newObject, dst(i, j));
			}
		}
	}
	return dst;
}

template<typename T>
class BlobReader {
public:
	virtual void readPixel(const T& pixel) = 0;
	virtual bool readPixel(const T& pixel, const T& parentPixel) = 0;
	virtual Blob<T> getBlob() = 0;
	virtual ~BlobReader() {};
};

template<typename PixelType, typename BlobReaderType, typename BoolConvertible>
std::tuple <Mat_<int>, std::map<int, Blob<PixelType>>>
regionGrowing(const Mat_<PixelType>& img, const Mat_<BoolConvertible>& seedPoints);
template<typename T>
void computeBlobs(std::map<int, Blob<T>>& blobs, Mat_<int>& labels, int background = -1);

class MonochromeRGBBlobReader : public BlobReader<Vec3b> {
private:
	const long	RA_TH = 60, GA_TH = 60, BA_TH = 60,
		R_TH = 5, G_TH = 5, B_TH = 5;
	long r, g, b;
	long area;
public:
	void readPixel(const Vec3b& pixel) override;
	bool readPixel(const Vec3b& pixel, const Vec3b& parentPixel) override;
	Blob<Vec3b> getBlob() override;
};

class MonochromeHSVBlobReader : public BlobReader<Vec3b> {
private:
	const long HA_TH = 60, SA_TH = 60, H_TH = 4, S_TH = 4;
	long h, s, v;
	long area;
public:
	void readPixel(const Vec3b& pixel) override;
	bool readPixel(const Vec3b& pixel, const Vec3b& parentPixel) override;
	Blob<Vec3b> getBlob() override;
};

class MonochromeLabBlobReader : public BlobReader<Vec3b> {
private:
	const long AA_TH = 17, BA_TH = 17, A_TH = 4, B_TH = 4;
	long l, a, b;
	long area;
public:
	void readPixel(const Vec3b& pixel) override;
	bool readPixel(const Vec3b& pixel, const Vec3b& parentPixel) override;
	Blob<Vec3b> getBlob() override;
};

#include <map>
#include <set>
inline bool similar(Vec3b a, Vec3b b, float similarity_th) {
	return fabs(a[0] - b[0]) <= similarity_th &&
		fabs(a[1] - b[1]) <= similarity_th &&
		fabs(a[2] - b[2]) <= similarity_th;
}

inline Vec3b bgr_to_hsv(Vec3b pixel) {
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
	Vec3b result;
	result[0] = H * 255 / 360;
	result[1] = S * 255;
	result[2] = V * 255;
	return result;
}
inline cv::Vec3b hsv_to_bgr(cv::Vec3b hsv_pixel) {
	cv::Mat_<cv::Vec3b> hsv_mat(1, 1, hsv_pixel);
	cv::Mat_<cv::Vec3b> bgr_mat;
	cv::cvtColor(hsv_mat, bgr_mat, cv::COLOR_HSV2BGR);
	return bgr_mat(0, 0);
}
inline cv::Vec3b lab_to_bgr(cv::Vec3b lab_pixel)
{
	cv::Mat lab_mat(1, 1, CV_8UC3);
	lab_mat.at<cv::Vec3b>(0, 0) = lab_pixel;
	cv::Mat bgr_mat;
	cv::cvtColor(lab_mat, bgr_mat, cv::COLOR_Lab2BGR);
	auto pixel = bgr_mat.at<cv::Vec3b>(0, 0);
	return { (uchar)pixel[0],  (uchar)pixel[1],  (uchar)pixel[2] };
}

template<typename PixelType, typename BlobReaderType, typename BoolConvertible>
std::tuple <Mat_<int>, std::map<int, Blob<PixelType>>>
regionGrowing(const Mat_<PixelType>& img, const Mat_<BoolConvertible>& seedPoints) {

	static_assert(std::is_base_of<BlobReader<PixelType>, BlobReaderType>::value, "Required BlobReader");

	static Point neighbours[] = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
	std::map<int, Blob<PixelType>> blobs;
	long background_area = img.rows * img.cols;
	int label = 0;
	Mat_<int> labels = Mat_<int>::zeros(img.rows, img.cols);
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (labels(i, j) == 0 && seedPoints(i, j)) {
				BlobReaderType reader;
				reader.readPixel(img(i, j));
				++label;
				std::queue<Point> q;
				labels(i, j) = label;
				q.push({ j, i });
				while (!q.empty()) {
					Point p = q.front();
					q.pop();
					for (int k = 0; k < 8; ++k) {
						Point neighbour = p + neighbours[k];
						if (inside(img, neighbour.y, neighbour.x) &&
							labels(neighbour) == 0 &&
							reader.readPixel(img(neighbour), img(p))
							) {
							labels(neighbour.y, neighbour.x) = label;
							q.push(neighbour);
						}
					}
				}
				auto blob = reader.getBlob();
				background_area -= blob.area;
				blobs.insert({ label, blob });
			}
		}
	}
	blobs.insert({ 0, {background_area,0,0, {255, 255, 255}} });
	return std::make_tuple(labels, blobs);
}

template<typename T>
void computeBlobs(std::map<int, Blob<T>>& blobs, Mat_<int>& labels, int background) {
	static Point neighbours[] = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
	std::set<int> processed;
	Mat_<uchar> visited = Mat_<uchar>::zeros(labels.rows, labels.cols);
	long background_area = labels.rows * labels.cols;
	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			int label = labels(i, j);
			visited(i, j) = 255;
			if (processed.count(label) == 0) {
				processed.insert(label);
				int area = 1;
				std::queue<Point> q;
				q.push({ j, i });
				while (!q.empty()) {
					Point p = q.front();
					q.pop();
					for (int k = 0; k < 8; ++k) {
						Point neighbour = p + neighbours[k];
						if (inside(labels, neighbour.y, neighbour.x) && visited(neighbour) == 0 && labels(neighbour) == label) {
							visited(neighbour) = 255;
							q.push(neighbour);
							++area;
						}
					}
				}
				background_area -= area;
				float p = perimeter(labels, label);
				auto& blob = blobs.find(label)->second;
				blob.area = area;
				blob.perimeter = p;
				blob.circularity = circularity(area, p);
			}
		}
	}
	blobs.insert({ 0, {background_area, 0, 0, {255, 255, 255}} });
}

void MonochromeLabBlobReader::readPixel(const Vec3b& pixel) {
	l = pixel[0];
	a = pixel[1];
	b = pixel[2];
	area = 1;
}
bool MonochromeLabBlobReader::readPixel(const Vec3b& pixel, const Vec3b& parentPixel) {
	double newl = pixel[0];
	double newa = pixel[1];
	double newb = pixel[2];
	double avga = a / area;
	double avgb = b / area;
	if (std::abs(avga - newa) > AA_TH ||
		std::abs(avgb - newb) > BA_TH ||
		std::abs(parentPixel[1] - pixel[1]) > A_TH ||
		std::abs(parentPixel[2] - pixel[2]) > B_TH) {
		return false;
	}
	l += newl;
	a += newa;
	b += newb;
	area++;
	return true;
}
Blob<Vec3b> MonochromeLabBlobReader::getBlob() {
	auto color = lab_to_bgr({ static_cast<uchar>(l / area),
												static_cast<uchar>(a / area),
												static_cast<uchar>(b / area) });
	return Blob<Vec3b>(area, 0, 0, color);
}

void MonochromeHSVBlobReader::readPixel(const Vec3b& pixel) {
	h = pixel[0];
	s = pixel[1];
	v = pixel[2];
	area = 1;
}
bool MonochromeHSVBlobReader::readPixel(const Vec3b& pixel, const Vec3b& parentPixel) {
	long newh = pixel[0];
	long news = pixel[1];
	long newv = pixel[2];
	long avgh = h / area;
	long avgs = s / area;
	if (std::abs(avgh - newh) > HA_TH ||
		std::abs(avgs - news) > SA_TH ||
		std::abs(parentPixel[0] - pixel[0]) > H_TH ||
		std::abs(parentPixel[1] - pixel[1]) > S_TH) {
		return false;
	}
	h += newh;
	s += news;
	v += newv;
	area++;
	return true;
}
Blob<Vec3b> MonochromeHSVBlobReader::getBlob() {
	return Blob<Vec3b>(area, 0, 0, hsv_to_bgr({ static_cast<uchar>(h / area),
									static_cast<uchar>(s / area),
									static_cast<uchar>(v / area) }));
}


void MonochromeRGBBlobReader::readPixel(const Vec3b& pixel) {
	b = pixel[0];
	g = pixel[1];
	r = pixel[2];
	area = 1;
}
bool MonochromeRGBBlobReader::readPixel(const Vec3b& pixel, const Vec3b& parentPixel) {
	long newr = pixel[2];
	long newg = pixel[1];
	long newb = pixel[0];
	long avgr = r / area;
	long avgg = g / area;
	long avgb = b / area;
	if (std::abs(avgr - newr) > RA_TH ||
		std::abs(avgg - newg) > GA_TH ||
		std::abs(avgb - newb) > BA_TH ||
		std::abs(parentPixel[0] - pixel[0]) > B_TH ||
		std::abs(parentPixel[1] - pixel[1]) > G_TH ||
		std::abs(parentPixel[2] - pixel[2]) > R_TH) {
		return false;
	}
	r += newr;
	g += newg;
	b += newb;
	area++;
	return true;
}
Blob<Vec3b> MonochromeRGBBlobReader::getBlob() {
	return Blob<Vec3b>(area, 0, 0, { static_cast<uchar>(b / area),
									static_cast<uchar>(g / area),
									static_cast<uchar>(r / area) });
}

Mat_<Vec3b> detectBalloons(Mat_<Vec3b> img) {
	Mat lab;
	cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
	//imshow("Initial image", img);
	auto seedPoints = map_pixels<Vec3b, uchar>(lab, [](Vec3b pixel) {
		return (pixel[0] > 30 && pixel[0] < 210) ? 255 : 0;
		});
	auto segmented_data = regionGrowing<Vec3b, MonochromeLabBlobReader, uchar>(lab, seedPoints);
	Mat_<int> segmented_image = std::get<0>(segmented_data);
	auto& blobs = std::get<1>(segmented_data);
	Mat_<int> objects = filter<int>(segmented_image, 0,
		[&blobs](int label) {
			int area = blobs.find(label)->second.area;
			if (area >= 1000) return true;
			return false;
		});

	Mat mat;
	objects.convertTo(mat, CV_32F);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
	Mat closed;
	morphologyEx(mat, closed, MORPH_CLOSE, kernel);
	closed.convertTo(objects, CV_32S);
	/*Mat imgsg = map_pixels<int, Vec3b>(objects, [&blobs](int label) {
		return blobs.find(label)->second.color;
		});*/
	//imshow("Segmentation", imgsg);
	computeBlobs(blobs, objects, 0);
	auto roundObjects = filter<int>(objects, 0,
		[&blobs](int label) {
			return blobs.find(label)->second.circularity > 0.65;
		});
	Mat finalImage = map_pixels<int, Vec3b>(roundObjects, [&blobs](int label) {
		return blobs.find(label)->second.color;
		});
	return finalImage;
}

void testDetectBalloons() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat image = imread(fname, IMREAD_COLOR);
		auto finalImage = detectBalloons(image);
		imshow("Balloons", finalImage);
		waitKey(0);
	}
}

template<typename B>
inline double iou(Mat_<B> x, Mat_<B> y) {
	long long _union = 0, _intersection = 0;
	for (auto x_it = x.begin(), y_it = y.begin();
		x_it != x.end() && y_it != y.end(); ++x_it, ++y_it) {
		if (*x_it & *y_it) {
			++_intersection;
		}
		if (*x_it | *y_it) {
			++_union;
		}
	}
	return static_cast<double>(_intersection) / _union;
}

void testIOU() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat image = imread(fname, IMREAD_COLOR);
		auto balloons = detectBalloons(image);
		auto test_subject = map_pixels<Vec3b, uchar>(balloons, [](Vec3b pixel) {
			return (pixel[0] != 255 || pixel[1] != 255 || pixel[2] != 255) ? 255 : 0;
			});
		const auto contours = getBalloons(fname);
		Mat_<uchar> test_target = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
		drawBalloons(test_target, contours);
		imshow("Test subject", test_subject);
		imshow("Test target", test_target);
		std::cout << iou(test_subject, test_target) << std::endl;
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Test data extraction\n");
		printf(" 2 - Detect Balloons\n");
		printf(" 3 - IOU\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testFindBalloons();
				break;
			case 2:
				testDetectBalloons();
				break;
			case 3:
				testIOU();
				break;
		}
	}
	while (op!=0);
	return 0;
}
