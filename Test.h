#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

class Balloon {
private:
	std::vector<cv::Point> points;
public:
	friend void drawBalloons(cv::Mat& img, std::vector<Balloon> baloons);
	inline void addPoint(int x, int y) {
		points.emplace_back(x, y);
	}
};
void drawBalloons(cv::Mat& img, std::vector<Balloon> balloons);
std::vector<Balloon> getBalloons(std::string imagePath);