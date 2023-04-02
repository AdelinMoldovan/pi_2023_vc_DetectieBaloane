#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

class Baloon {
private:
	std::vector<cv::Point> points;
public:
	friend void drawBaloons(cv::Mat& img, std::vector<Baloon> baloons);
	inline void addPoint(int x, int y) {
		points.emplace_back(x, y);
	}
};
void drawBaloons(cv::Mat& img, std::vector<Baloon> baloons);
std::vector<Baloon> getBaloons(std::string imagePath);