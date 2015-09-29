/**
 * @author VaL Doroshchuk <valbok@gmail.com>
 * @created 26 Sep 2015
 */

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string>

using namespace cv;
using namespace std;

typedef vector<vector<Point> > TPoints;
void showHelp(const char *appName) {
    cerr << "Searches for geometrical shapes (circle, triangle, rectangle) within any image.\n" <<
        "Usage: " << appName << " filename\n" <<
        "Using OpenCV version " << CV_VERSION << "\n";
}

/**
 * Finds a cosine of angle between vectors
 * from pt0->pt1 and from pt0->pt2
 */
double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/**
 * Returns sequence of squares detected in the image.
 */
void find(const Mat& image, TPoints& shapes) {
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // Down-scale and upscale the image to filter out the noise.
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    TPoints contours;
    const int thresh = 50, N = 11;

    // Find squares in every color plane of the image.
    for (unsigned c = 0; c < 3; ++c) {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // Try several threshold levels.
        for (unsigned l = 0; l < N; ++l) {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0) {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            } else {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // Find contours
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;
            for (unsigned i = 0; i < contours.size(); ++i) {
                // Approximate contour with accuracy proportional
                // to the contour perimeter.
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // Skip small or non-convex objects.
                if (fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx))
                    continue;

                // Number of vertices.
                int vtc = approx.size();
                if (vtc == 3) {
                    shapes.push_back(approx); // Triangle
                } else if (vtc >= 4 && vtc <= 6) {
                    // Get the cosines of all corners
                    vector<double> cos;
                    for (int j = 2; j < vtc + 1; ++j) {
                        cos.push_back(angle(approx[j % vtc], approx[j - 2], approx[j - 1]));
                    }

                    // Sort ascending the cosine values.
                    sort(cos.begin(), cos.end());

                    // Get the lowest and the highest cosine.
                    double mincos = cos.front();
                    double maxcos = cos.back();

                    // Use the degrees obtained above and the number of vertices
                    // to determine the shape of the contour.
                    if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3) {
                        shapes.push_back(approx); // Rect
                    } else if (vtc == 5 && mincos >= -0.35 && maxcos <= -0.21) {
                        shapes.push_back(approx); // Penta
                    } else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45) {
                        shapes.push_back(approx); // Hexa
                    }
                } else {
                    double area = contourArea(contours[i]);
                    Rect r = boundingRect(contours[i]);
                    int radius = r.width / 2;
                    if (abs(1 - ((double)r.width / r.height)) <= 0.3 &&
                        abs(1 - (area / (CV_PI * pow(radius, 2)))) <= 0.2) {
                        shapes.push_back(approx); // Circle
                    }
                }
            }
        }
    }
}

/**
 * Drwas squares in the image.
 */
void draw(Mat& image, const TPoints& squares) {
    for(size_t i = 0; i < squares.size(); ++i) {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 1, CV_AA);
    }

    imshow("Geometrical shapes", image);
    waitKey(0);
}

int main(int argc, const char** argv) {
    string path;
    if (argc > 1) {
        path = argv[1];
    }
    if (path.empty()) {
        showHelp(argv[0]);
        return 1;
    }

    TPoints shapes;

    Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        cerr << "Couldn't load image " << path << endl;
        return 1;
    }

    find(image, shapes);
    draw(image, shapes);

    return 0;
}
