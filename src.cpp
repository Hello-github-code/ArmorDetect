#include <opencv2/opencv.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

using namespace cv;
using namespace std;

struct LightInfo {
    RotatedRect boundingRect;  // 外包矩形
    float angle;               // 角度
    float aspectRatio;         // 高宽比
    float area;                // 面积
    float height;              // 高度
    Point2f center;            // 中心
    
    LightInfo(const RotatedRect& rotRect) // 构造函数
        : boundingRect(rotRect), angle(rotRect.angle), 
        aspectRatio(rotRect.size.height / rotRect.size.width), 
        area(rotRect.size.area()), height(rotRect.size.height), 
        center(rotRect.center) {}
};

float distance(Point2f center1, Point2f center2) {
    float x_diff = abs(center1.x - center2.x);
    float y_diff = abs(center1.y - center2.y);
    float dis_sqr = pow(x_diff, 2) + pow(y_diff, 2);
    float dis = sqrt(dis_sqr);
    return dis; 
}

// 获取装甲板的最小外接旋转矩形
RotatedRect getCombinedRect(const LightInfo& light1, const LightInfo& light2) {
    // 合并成八个点
    Point2f points1[4], points2[4];
    light1.boundingRect.points(points1);
    light2.boundingRect.points(points2);
    vector<Point2f> combinedPoints(points1, points1 + sizeof(points1) / sizeof(points1[0]));
    combinedPoints.insert(combinedPoints.end(), points2, points2 + sizeof(points2) / sizeof(points2[0]));

    // 寻找面积最大的四个顶点
    vector<Point2f> maxAreaPoints;
    float maxArea = 0;

    for (size_t i = 0; i < combinedPoints.size(); i++) {
        for (size_t j = i + 1; j < combinedPoints.size(); j++) {
            for (size_t k = j + 1; k < combinedPoints.size(); k++) {
                for (size_t l = k + 1; l < combinedPoints.size(); l++) {
                    vector<Point2f> currentPoints = {combinedPoints[i], combinedPoints[j], combinedPoints[k], combinedPoints[l]};
                    float currentArea = contourArea(currentPoints);
                    if (currentArea > maxArea) {
                        maxArea = currentArea;
                        maxAreaPoints = currentPoints;
                    }
                }
            }
        }
    }

    RotatedRect combinedRect = minAreaRect(maxAreaPoints);
    return combinedRect;
}




int main() {

    YAML::Node config = YAML::LoadFile("../config.yml");
    string videoPath = config["video_path"].as<string>(); // 视频路径
    string lightColor = config["light_color"].as<string>(); // 识别颜色

    float light_min_area = config["light_min_area"].as<float>(); // 灯条最小面积
    float light_max_angle = config["light_max_angle"].as<float>(); // 灯条最大角度
    float light_max_ratio = config["light_max_ratio"].as<float>(); // 灯条最大高宽比
    float light_min_ratio = config["light_min_ratio"].as<float>(); // 灯条最小高宽比
    float light_contour_min_solidity = config["light_contour_min_solidity"].as<float>(); // 灯条最小实心程度
    float light_extend_ratio = config["light_extend_ratio"].as<float>(); // 灯条扩大倍数
    
    float lights_angle_differ = config["lights_angle_differ"].as<float>(); // 灯条最大角度差
    float lights_height_diff_ratio = config["lights_height_diff_ratio"].as<float>(); // 灯条最大高度差比率
    float lights_max_y_diff_ratio = config["lights_max_y_diff_ratio"].as<float>(); // 灯条最大y差比率
    float lights_min_x_diff_ratio = config["lights_min_x_diff_ratio"].as<float>(); // 灯条最小x差比率
    
    float armor_max_ratio = config["armor_max_ratio"].as<float>(); // 装甲板最大比率
    float armor_min_ratio = config["armor_min_ratio"].as<float>(); // 装甲板最小比率
    float armor_angle_limit = config["armor_angle_limit"].as<float>(); // 装甲板最大偏转

    // 打开视频文件
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    int x = 1; // 计数
    while (true) {
        // 读取图像
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        double fps = cap.get(CAP_PROP_FPS);

        // 创建灯条容器
        vector<LightInfo> lights;

        // 预处理: 从BGR颜色空间到HSV颜色空间的转换
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV); 
        // 阈值化处理: 提取出指定颜色范围内的部分
        Scalar lower, upper;
        if (lightColor == "blue") {
            lower = Scalar(100, 100, 100);
            upper = Scalar(140, 255, 255);
        } else if (lightColor == "red") {
            lower = Scalar(0, 100, 100);
            upper = Scalar(10, 255, 255);
        } 
        Mat mask;
        inRange(hsv, lower, upper, mask); 
        // 膨胀
        Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
        dilate(mask, mask, element);
        // 显示二值化图像
        imshow("二值化", mask);

        // 识别灯条
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            // 面积太小的不要
	        float lightContourArea = contourArea(contour);
            if(lightContourArea < light_min_area) continue;  

            // 获取最小外接旋转矩形
            RotatedRect lightRect = minAreaRect(contour);

            // 矫正
            if (lightRect.size.width > lightRect.size.height) {
			    lightRect.angle += 90;
			    float t = lightRect.size.width;
			    lightRect.size.width = lightRect.size.height;
			    lightRect.size.height = t;
		    } 
            
            // 角度筛选
            if (abs(lightRect.angle) > light_max_angle) continue;

            // 高比宽、凸度筛选灯条
            if( lightRect.size.height / lightRect.size.width > light_max_ratio ||
                lightRect.size.height / lightRect.size.width < light_min_ratio ||
	            lightContourArea / lightRect.size.area() < light_contour_min_solidity
            ) continue;

            //对灯条范围适当扩大
	        lightRect.size.width *= light_extend_ratio;
	        lightRect.size.height *= light_extend_ratio;

            // // 提取感兴趣部分
            // Rect light = lightRect.boundingRect();
            // const Rect srcBound(Point(0, 0), frame.size());
            // light &= srcBound;

            lights.push_back(LightInfo(lightRect));
        }

        // 排序
        sort(lights.begin(), lights.end(), [](const LightInfo& ld1, const LightInfo& ld2) {
	        return ld1.boundingRect.center.x < ld2.boundingRect.center.x;
        });

        // 绘制灯条
        Scalar color;
        if (lightColor == "blue") {
            color = Scalar(255, 0, 0);
        } else if (lightColor == "red") {
            color = Scalar(0, 0, 255);
        } 
        for (const auto& light : lights) {
            Point2f vertices[4];
            light.boundingRect.points(vertices);
            for (int j = 0; j < 4; j++) {
                line(frame, vertices[j], vertices[(j + 1) % 4], color, 2);
            }
        }

        // 识别装甲板
        vector<RotatedRect> armors;
        for(size_t i = 0; i < lights.size(); i++) {
	        for(size_t j = i + 1; j < lights.size(); j++) {
		        const LightInfo& leftLight  = lights[i];
		        const LightInfo& rightLight = lights[j];

                // 角度差
                float angle_diff = abs(leftLight.angle - rightLight.angle);
                // 高度差比率
                float LenDiff_ratio = abs(leftLight.height - rightLight.height) / max(leftLight.height, rightLight.height);
                // 左右灯条中心距离
		        float dis = distance(leftLight.center, rightLight.center);
                // 左右灯条高度平均值
                float Len = (leftLight.height + rightLight.height) / 2;
                // 左右灯条中心点y的差值
                float yDiff = abs(leftLight.center.y - rightLight.center.y);
                // y差比率
                float yDiff_ratio = yDiff / Len;
                // 左右灯条中心点x的差值
                float xDiff = abs(leftLight.center.x - rightLight.center.x);
                // x差比率
                float xDiff_ratio = xDiff / Len;

                if (angle_diff > lights_angle_differ ||
		           LenDiff_ratio > lights_height_diff_ratio ||
                   yDiff_ratio > lights_max_y_diff_ratio ||
		           xDiff_ratio < lights_min_x_diff_ratio) continue;

                // 装甲板比值
                float ratio = dis / Len;

                if (ratio > armor_max_ratio ||
		            ratio < armor_min_ratio) continue;
                
                // 创建装甲板
                RotatedRect armor = getCombinedRect(lights[i], lights[j]);
                
                // 矫正
                if (armor.size.width < armor.size.height) {
			        armor.angle += 90;
			        float t = armor.size.width;
			        armor.size.width = armor.size.height;
			        armor.size.height = t;
		        }

                if (fabs(armor.angle) > armor_angle_limit) continue;

                armors.push_back(armor);

		        break;
	        }
        }

        // 按照面积大小对装甲板进行排序
        sort(armors.begin(), armors.end(), [](const RotatedRect& rect1, const RotatedRect& rect2) {
            return rect1.size.area() > rect2.size.area();
        });
        
        // 绘制装甲板
        Scalar color2;
        if (lightColor == "blue") {
            color2 = Scalar(0, 255, 0);
        } else if (lightColor == "red") {    
            color2 = Scalar(0, 255, 0);
        } 
        for (const auto& armor : armors) {
                Point2f vertices[4];
                armor.points(vertices);
                for (int j = 0; j < 4; j++) {
                    line(frame, vertices[j], vertices[(j + 1) % 4], color2, 2);
                }
                cout << "第几帧: " << x <<endl;
                cout << "装甲板中心点: " << armor.center << endl;
                cout << "装甲板偏转角度: " << armor.angle << endl;
                cout << "装甲板面积: " << armor.size.width * armor.size.height << endl << endl;
        }

        // 在图像上输出帧率
        putText(frame, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // 显示图像
        namedWindow("Armor Recognition", WINDOW_NORMAL);
        resizeWindow("Armor Recognition", 2400, 2000);
        imshow("Armor Recognition", frame);

        x++;

        if (waitKey(0) == 27) {
            break;
        }
    }

    cap.release();

    return 0;
}























