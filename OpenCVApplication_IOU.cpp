#include "stdafx.h"
#include "common.h"
#include "Test.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <set>
#include <random>
#include <functional>
#include <cmath>

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("opened image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i, j) = (r + g + b) / 3;
			}
		}

		imshow("original image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

Mat testFindBalloons(Mat& src,char* fname)
{
	const auto baloons = getBalloons(fname);
	drawBalloons(src, baloons);
	imshow("opened image", src);
	return src;
}
void detectLines() {
	// Încarcă imaginea
	Mat image;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		image = imread(fname, IMREAD_ANYCOLOR);
		// Aplică filtrul Gaussian
		Mat blurred;
		GaussianBlur(image, blurred, Size(5, 5), 0);

		// Detectează marginile cu algoritmul Canny
		Mat edges;
		Canny(blurred, edges, 50, 150);

		// Afișează imaginea inițială și marginile detectate
		imshow("Imaginea initiala", image);
		imshow("Marginile detectate", edges);
		waitKey(0);
	}
}

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

inline boolean inside(Mat img, int r, int c) {
	return (r < img.rows) && (r >= 0) && (c >= 0) && (c < img.cols);
}
class Blob {
public:
	int area;
	float perimeter;
	float circularity;
	Blob(int area, float perimeter, float circularity) :
		area(area), perimeter(perimeter), circularity(circularity) {};
};
inline boolean similar(Vec3b a, Vec3b b, float similarity_th) {
	return fabs(a[0] - b[0]) <= similarity_th &&
		fabs(a[1] - b[1]) <= similarity_th &&
		fabs(a[2] - b[2]) <= similarity_th;
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
inline double circularity(int area, double perimeter) {
	return PI * 4 * area / (perimeter * perimeter);
}
std::tuple <Mat_<int>, std::map<int, Blob>> regionGrowing(Mat_<Vec3b> img, const float tolerance) {
	static Point neighbours[] = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
	std::map<int, Blob> blobs;
	float similarity_th = tolerance / 100.0 * 255.0;
	int label = 0;
	Mat_<int> labels = Mat_<int>::zeros(img.rows, img.cols);
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (labels(i, j) == 0) {
				++label;
				int area = 1;
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
							similar(img(neighbour), img(p), similarity_th)) {
							labels(neighbour.y, neighbour.x) = label;
							q.push(neighbour);
							++area;
						}
					}
				}
				blobs.insert({ label, Blob(area, 0,  0) });
			}
		}
	}
	return std::make_tuple(labels, blobs);
}

Vec3b randomColor() {
	static std::random_device rd;
	static std::mt19937 generator(rd());
	static std::uniform_int_distribution<> distribution(0, 255);
	return Vec3b(distribution(generator), distribution(generator), distribution(generator));
}
void computeBlobs(std::map<int, Blob>& blobs, Mat_<int>& labels, int background = -1) {
	static Point neighbours[] = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
	std::set<int> processed;
	//processed.insert(background);
	Mat_<uchar> visited = Mat_<uchar>::zeros(labels.rows, labels.cols);
	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			int label = labels(i, j);
			visited(i, j) = 255;
			if (processed.count(label) == 0) {
				//std::cout << label << std::endl;
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
				float p = perimeter(labels, label);
				auto& blob = blobs.find(label)->second;
				blob.area = area;
				blob.perimeter = p;
				blob.circularity = circularity(area, p);
			}
		}
	}
}
bool operator<(const cv::Vec3b& lhs, const cv::Vec3b& rhs) {
	return lhs[0] < rhs[0] ||
		(lhs[0] == rhs[0] && lhs[1] < rhs[1]) ||
		(lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] < rhs[2]);
}
struct Vec3bComparator {
	bool operator()(const Vec3b lhs, const Vec3b rhs) const {
		return lhs < rhs;
	}
};
Mat_<Vec3b> colorCodeLabels(Mat_<int> labels) {
	static const Vec3b WHITE = Vec3b(255, 255, 255);
	std::map<int, Vec3b> labelMap;
	std::set<Vec3b, Vec3bComparator> colors;
	Mat_<Vec3b> colorMap = Mat_<Vec3b>(labels.rows, labels.cols, { 255, 255, 255 });
	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			int label = labels(i, j);
			if (label != 0) {
				if (labelMap.count(label) == 0) {
					Vec3b color = randomColor();
					while (color != WHITE && colors.count(color) != 0) {
						color = randomColor();
					}
					labelMap.insert({ label, color });
					colors.insert(color);
					colorMap(i, j) = color;
				}
				else {
					colorMap(i, j) = labelMap[label];
				}
			}
		}
	}
	return colorMap;
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
Mat_<T> map(
	const Mat_<T>& img,
	T background,
	std::function<Mat_<T>(Mat_<T>)> map_object,
	std::function<void(Mat_<T>&, const Mat_<T>, const T)> mix) {

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
cv::Mat detectBalloons(const cv::Mat& clr,const cv::Mat& img) 
{
	imshow("Initial image", clr);
	auto segmented_data = regionGrowing(clr, 2);
	Mat_<int> segmented_image = std::get<0>(segmented_data);
	auto& blobs = std::get<1>(segmented_data);
	Mat_<int> objects = filter<int>(segmented_image, 0,
		[&blobs](int label) {
			int area = blobs.find(label)->second.area;
	if (area >= 1000) return true;
	return false;
		});
	Mat imgsg = colorCodeLabels(objects);
	imshow("Segmentation", imgsg);

	Mat mat;
	objects.convertTo(mat, CV_32F);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	Mat closed;
	morphologyEx(mat, closed, MORPH_CLOSE, kernel);
	closed.convertTo(objects, CV_32S);
	blobs.insert({ 0, {0,0,0} });
	computeBlobs(blobs, objects, 0);
	auto roundObjects = filter<int>(objects, 0,
		[&blobs](int label) {
			return blobs.find(label)->second.circularity > 0.65;
		});
	Mat finalImage = colorCodeLabels(roundObjects);
	Mat binaryImage = finalImage.clone();
	for (int i = 0; i < binaryImage.rows; i++)
		for (int j = 0; j < binaryImage.cols; j++)
		{
			if (binaryImage.at<Vec3b>(i, j) != Vec3b({ 255, 255, 255 }))
				binaryImage.at<Vec3b>(i, j) = Vec3b({ 0,0,0 });
		}
	
	return binaryImage;
	
}

double truncateToTwoDecimals(double value) {
	double multiplier = std::pow(10.0, 3); // Set the multiplier for two decimal places
	double truncatedValue = std::floor(value * multiplier) / multiplier;
	return truncatedValue;
}

void TestIOU() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		const Vec3b BLACK = Vec3b({ 0,0,0 });
		Mat clr = imread(fname, IMREAD_COLOR);
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		
		Mat detectedBaloons = detectBalloons(clr, img);
		imshow("Detected Baloons", detectedBaloons);

		Mat testBaloons = testFindBalloons(img,fname);
		Mat testBaloons3b;
		cvtColor(testBaloons, testBaloons3b, COLOR_GRAY2BGR);

		int _union = 0;
		int _intersection = 0;
		for (int i = 0; i < detectedBaloons.rows - 1; i++) {
			for (int j = 0; j < detectedBaloons.cols - 1; j++) {
				if (detectedBaloons.at<Vec3b>(i, j) == BLACK && testBaloons3b.at<Vec3b>(i, j) == BLACK) {
					_intersection++;
				}
				if (detectedBaloons.at<Vec3b>(i, j) == BLACK || testBaloons3b.at<Vec3b>(i, j) == BLACK) {
					_union++;
				}
			}
		}
		double iou = truncateToTwoDecimals((double)_intersection / _union) * 100;
		std::cout << "IOU: ";
		std::cout << iou << "%";
		waitKey();
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
		printf(" 1 - Basic image opening...\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Color to Gray\n");
		printf(" 4 - Test IOU (calculated over test set)\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testColor2Gray();
			break;
		//case 4:
		//	testFindBalloons();
		//	break;
		case 4:
			TestIOU();
			break;
		}
	} while (op != 0);
	return 0;
}