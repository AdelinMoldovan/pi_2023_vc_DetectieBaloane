#include "stdafx.h"

#include "test.h"
#include "json/json.h"
#include <iostream>
#include <fstream>

Json::Value getJson(std::string filename) {
	std::ifstream file(filename);
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	if (!Json::parseFromStream(builder, file, &root, &errors)) {
		std::cerr << "Error parsing JSON data: " << errors << std::endl;
	}
	file.close();
	return root;
}

void drawBalloons(cv::Mat& img, std::vector<Balloon> balloons) {
	std::vector<std::vector<cv::Point>> contours;
	for (const auto& balloon : balloons) {
		contours.push_back(balloon.points);
	}
	cv::fillPoly(img, contours, cv::Scalar(255, 255, 255));
}

std::vector<Balloon> getBalloons(std::string imagePath) {
	size_t lastSeparatorPos = imagePath.find_last_of("\\/");
	std::string imageName = imagePath.substr(lastSeparatorPos + 1);
	std::string jsonName = imagePath.substr(0, lastSeparatorPos + 1) + "via_region_data.json";
	auto root = getJson(jsonName);
	Json::Value image;
	for (const std::string& key : root.getMemberNames()) {
		if (key.rfind(imageName, 0) == 0)
			image = root[key];
	}
	std::vector<Balloon> balloons;
	for (const std::string& key : image["regions"].getMemberNames()) {
		auto region = image["regions"][key]["shape_attributes"];
		Balloon balloon;
		const auto& xArray = region["all_points_x"];
		const auto& yArray = region["all_points_y"];
		for (Json::ArrayIndex i = 0; i < xArray.size(); ++i) {
			int x = xArray[i].asInt();
			int y = yArray[i].asInt();
			balloon.addPoint(x, y);
		}
		balloons.push_back(balloon);
	}
	return balloons;
}
