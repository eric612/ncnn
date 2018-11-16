// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include <fstream>
#include <sstream>
#include <string>
#include <libgen.h>
using namespace std;
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};



static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects,int coordinate_type = 0)
{
    ncnn::Net yolov3;

    // original pretrained model from https://github.com/eric612/Caffe-YOLOv3-Windows
    // yolov3_deploy.prototxt
	// https://github.com/eric612/MobileNet-YOLO/blob/master/models/darknet_yolov3/yolov3.prototxt
    // https://drive.google.com/file/d/12nLE6GtmwZxDiulwdEmB3Ovj5xx18Nnh/view
    yolov3.load_param("yolo.param");
    yolov3.load_model("yolo.bin");

    // https://github.com/eric612/MobileNet-YOLO
    // https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov3/mobilenet_yolov3_lite_deploy.prototxt
    // https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov3/mobilenet_yolov3_lite_deploy.caffemodel


    const int target_size = 416;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    // the Caffe-yolov3-Windows style
    // X' = X * scale - mean
    const float mean_vals[3] = {0.5f, 0.5f, 0.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(0, norm_vals);
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

//     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
		if (coordinate_type == 1) {
			object.rect.x = values[2] * img_w + 1;
			object.rect.y = values[3] * img_h + 1;
			object.rect.width = values[4] * img_w + 1;
			object.rect.height = values[5] * img_h + 1;
		}
		else {
			object.rect.x = values[2] * img_w;
			object.rect.y = values[3] * img_h;
			object.rect.width = values[4] * img_w - object.rect.x;
			object.rect.height = values[5] * img_h - object.rect.y;
		}

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects,int waiting_time = 0)
{
	static const char* class_names[] = { "background",
		"aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(waiting_time);
	cv::imwrite("img.jpg",image);
}
string getFileName(const string& s) {

	char sep = '/';

#ifdef _WIN32
	sep = '\\';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		string s2 = s.substr(i + 1, s.length() - i);

		return(s2.substr(0, s2.length() - 4));
	}

	return("");
}
void print_detector_detections(FILE **fps, char *id, std::vector<Object> dets, int total, int classes, int w, int h)
{
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = dets[i].rect.x ;
		float xmax = dets[i].rect.width ;
		float ymin = dets[i].rect.y ;
		float ymax = dets[i].rect.height ;

		if (xmin < 1) xmin = 1;
		if (ymin < 1) ymin = 1;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j) {
			if (j == dets[i].label - 1) {
				if (dets[i].prob) 
					fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob,xmin, ymin, xmax, ymax);
				//if (dets[i].prob)
				//	printf("%s %f %f %f %f %f %d\n", id, dets[i].prob, xmin, ymin, xmax, ymax, dets[i].label);
			}
		}
	}
}
void valid()
{
	static const char* class_names[] = {
		"aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor" };

	char buf[1000];
	sprintf(buf, "%s/*.jpg", "images//");
	cv::String path(buf); //select only jpg
	vector<cv::String> fn;
	vector<cv::Mat> data;
	cv::glob(path, fn, true); // recurse
	FILE *fp = 0;
	FILE **fps = 0;
	char *outfile = "comp4_det_test_";
	int classes = 20;
	//int size = sizeof(FILE *);
	fps = (FILE **)calloc(classes, sizeof(FILE *));
	char buff[1024];
	//mkdir("results\");
	for (int j = 0; j < classes; ++j) {
		snprintf(buff, 1024, "%s/%s%s.txt", "results", outfile, class_names[j]);
		fps[j] = fopen(buff, "w");
	}
	std::ifstream infile("2007_test.txt");
	std::string str;
	vector<char*> list;
	vector<char*> id_list;
	int idx = 0;
	while (std::getline(infile, str))
	{
		const char *cstr = str.c_str();
		string id = getFileName(str);

		char *str = new char[256];
		sprintf(str, "%s", cstr);
		list.push_back(str);
		char *str2 = new char[256];
		sprintf(str2, "%s", id.c_str());
		id_list.push_back(str2);
		idx++;
	}
	int volatile count = 0;
#pragma omp parallel for num_threads(4)
	for(int i=0;i<idx;i++)
	{
		char *cstr = list[i];

		
		cv::Mat img = cv::imread(cstr);
		if (img.empty())
		{
			fprintf(stderr, "cv::imread failed\n");
			continue;
		}
		std::vector<Object> objects;
		detect_yolov3(img, objects, 1);
		//printf("%d %d\n", img.cols, img.rows);
		print_detector_detections(fps, id_list[i], objects, objects.size(), 20, img.cols, img.rows);
		count++;
		if (count % 4 == 0) {
			printf("%d\n", count);
		}
	}
	/*for (size_t k = 0; k < fn.size(); ++k)
	{

		//draw_objects(img, objects,1000);
		//cv::waitKey(1000);
	}*/
}
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
	int res = -1;

	res = strcmp(argv[1], "valid");
	if (res == 0) {
		printf("valid start\n");
		valid();
		return 0;
	}
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov3(m, objects);

    draw_objects(m, objects);

    return 0;
}
