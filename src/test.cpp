#include "lbf/lbf.hpp"

#include <cstdio>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;
using namespace lbf;

//---add by DC------------------2017.8.4-pm
Mat getinitShape(vector<Mat> &gt_shapes, vector<BBox> &bboxes);

// dirty but works
void parseTxt(string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes);

int test(void) {
    Config &config = Config::GetInstance();

    LbfCascador lbf_cascador;
    FILE *fd = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(fd);
    fclose(fd);

    LOG("Load test data from %s", config.dataset.c_str());
    string txt = config.dataset + "/test.txt";
    vector<Mat> imgs, gt_shapes;
    vector<BBox> bboxes;
    parseTxt(txt, imgs, gt_shapes, bboxes);

    int N = imgs.size();
    lbf_cascador.Test(imgs, gt_shapes, bboxes);

    return 0;
}

int run(void) {
    Config &config = Config::GetInstance();
    FILE *fd = fopen((config.dataset + "/test.txt").c_str(), "r");
    assert(fd);
    int N;
    int landmark_n = config.landmark_n;
    fscanf(fd, "%d", &N);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);

    LbfCascador lbf_cascador;
    FILE *model = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(model);
    fclose(model);

    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path);

        // crop img
        double x_min, y_min, x_max, y_max;
        x_min = *min_element(x.begin(), x.end());
        x_max = *max_element(x.begin(), x.end());
        y_min = *min_element(y.begin(), y.end());
        y_max = *max_element(y.begin(), y.end());
        x_min = max(0., x_min - bbox[2] / 2);
        x_max = min(img.cols - 1., x_max + bbox[2] / 2);
        y_min = max(0., y_min - bbox[3] / 2);
        y_max = min(img.rows - 1., y_max + bbox[3] / 2);
        double x_, y_, w_, h_;
        x_ = x_min; y_ = y_min;
        w_ = x_max - x_min; h_ = y_max - y_min;
        BBox bbox_(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        Rect roi(x_, y_, w_, h_);
        img = img(roi).clone();

        Mat gray;
        cvtColor(img, gray, CV_BGR2GRAY);
        LOG("Run %s", img_path);
        Mat shape = lbf_cascador.Predict(gray, bbox_);

        //--------------add by DC: show init landmark

        

        Config &config = Config::GetInstance();
        LOG("Load train data from %s", config.dataset.c_str());
        string txt = config.dataset + "/train.txt";
        vector<Mat> imgs_, gt_shapes_;
        vector<BBox> bboxes_;
        parseTxt(txt, imgs_, gt_shapes_, bboxes_);

        
        LOG("get Mean Shape");
        std::cout << std::endl;
        std::cout << gt_shapes_[0].rows << std::endl; 
        std::cout << gt_shapes_[0].cols << std::endl; 
        std::cout << gt_shapes_[0].dims << std::endl; 
        Mat mean_shape = getinitShape(gt_shapes_, bboxes_);

        std::cout << "Meanshape = "<< std::endl << " "  << mean_shape << std::endl;
        
        
        
        Mat init_img = imread(img_path);
        Mat init_gray;
        cvtColor(init_img, init_gray, CV_BGR2GRAY);
        init_img = drawShapeInImage(init_img, mean_shape, bbox_);



        
        img = drawShapeInImage(img, shape, bbox_);
        std::cout << "shape = "<< std::endl << " "  << shape << std::endl;


        namedWindow("init", CV_WINDOW_NORMAL);
        imshow("init", init_img);
        imwrite("../Results/init.jpg", init_img);

        namedWindow( "result", CV_WINDOW_NORMAL );
        imshow("result", img);
        imwrite( "../Results/result.jpg", img);
        //cvSaveImage("../result.jpg",img);
        waitKey(0);
    }
    fclose(fd);
    return 0;
}











//----------------------------------------------------------------------------DC's function
Mat getinitShape(vector<Mat> &gt_shapes, vector<BBox> &bboxes)
{
    int N = gt_shapes.size();
    Mat mean_shape = Mat::zeros(gt_shapes[0].rows, 2, CV_64FC1);
    for (int i = 0; i < N; i++) {
        mean_shape += gt_shapes[i];
    }
    mean_shape /= N;
    return mean_shape;
}