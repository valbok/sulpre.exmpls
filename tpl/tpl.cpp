/**
 * @author VaL Doroshchuk <valbok@gmail.com>
 * @created 29 Sep 2015
 */

/**
 * Command line application that takes two input images: Needle-image and haystack-image. 
 * The algorithm tries to find areas in haystack-image that resemble needle-image. 
 * In other words searches small image in big image. Very simplified method.
 * And returns result how well needle image can match in haystack.
 */

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <climits>
#include <cstdlib>
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;

void showHelp(const char *appName) {
    cerr << "Searches needle image in haystack and returns result match value from 0 to 1.\n" <<
        "Usage: " << appName << " haystack needle\n" <<
        "Using OpenCV version " << CV_VERSION << "\n";
}

namespace tpl {

/**
 * Creates an integral image \a sum.
 * @param Source image
 * @param Integral image
 */
void integral(const Mat& src, Mat& sum) {
    sum.create(src.rows, src.cols, CV_MAKETYPE(CV_32S, src.channels()));
    for (int y = 0; y < src.rows; ++y) {
        int s = 0;
        for (int x = 0; x < src.cols; ++x) {
            Vec3b c = src.at<Vec3b>(Point(x, y));
            s += c.val[0] + c.val[1] + c.val[2];
            int a = y > 0 ? sum.at<int>(Point(x, y - 1)) : 0;
            sum.at<int>(Point(x, y)) = s + a;
        }
    }
}

} // namespace tpl

/**
 * Candidate item to compare integral images.
 */
struct item {
    /**
     * Difference between needle integral summ and piece of the same size on haystack.
     */
    int diff;

    /**
     * Coordinates.
     */
    int x;
    int y;

    /**
     * Sum of piece from haystack.
     */
    int sum;
};


int main(int argc, const char** argv) {
    string haystack_path, needle_path;
    if (argc > 2) {
        haystack_path = argv[1];
        needle_path = argv[2];
    }
    if (haystack_path.empty()) {
        showHelp(argv[0]);
        return 1;
    }

    Mat haystack = imread(haystack_path, CV_LOAD_IMAGE_COLOR);
    Mat needle = imread(needle_path, CV_LOAD_IMAGE_COLOR);
    if (haystack.empty() || needle.empty()) {
        cerr << "Couldn't load images!" << endl;
        return 1;
    }

    Mat haystack_sum;
    tpl::integral(haystack, haystack_sum);

    Mat needle_sum;
    tpl::integral(needle, needle_sum);

    int ns = needle_sum.at<int>(needle_sum.rows - 1, needle_sum.cols - 1);

    // Contains candidate results.
    deque<item> deq;

    for (int y = 0; y < haystack_sum.rows - needle_sum.rows; ++y) {
        for (int x = 0; x < haystack_sum.cols - needle_sum.cols; ++x) {
            int s = haystack_sum.at<int>(y, x) +
                haystack_sum.at<int>(y + needle_sum.rows, x + needle_sum.cols) -
                haystack_sum.at<int>(y, x + needle_sum.cols) -
                haystack_sum.at<int>(y + needle_sum.rows, x);

            int d = abs(s - ns);
            item itm = {d, x, y, s};
            if (deq.size() > 0) {
                for (auto it = deq.begin(); it != deq.end(); ++it) {
                    // Need to store candidates to check further.
                    if (d < it->diff) {
                        deq.insert(it, itm);
                        break;
                    }
                }
            } else {
                deq.push_front(itm);
            }

            if (deq.size() > 50) {
                deq.pop_back();
            }
        }
    }

    const int nc = 3; // Number of channels.
    const int byte = 255; // One byte.
    const int max = needle.rows * needle.cols * byte * nc; // Maximum value that can be in comparing by brute force.

    float result = !deq.empty() && deq[0].diff == 0 ? 1 : 0;
    int min = INT_MAX;
    // Result point.
    int rx = !deq.empty() && deq[0].diff == 0 ? deq[0].x : -1;
    int ry = !deq.empty() && deq[0].diff == 0 ? deq[0].y : -1;

    // If perfect result has not been found -> need to find it by brute force.
    if (result != 1) {
        for (int d = 0; d < deq.size(); ++d) {
            unsigned long long s = 0;
            for (int j = 0; j < needle.rows; ++j) {
                for (int i = 0; i < needle.cols; ++i) {
                    Vec3b c1 = haystack.at<Vec3b>(Point(deq[d].x + i, deq[d].y + j));
                    Vec3b c2 = needle.at<Vec3b>(Point(i, j));
                    s += abs(c1.val[0] - c2.val[0]);
                    s += abs(c1.val[1] - c2.val[1]);
                    s += abs(c1.val[2] - c2.val[2]);
                }
            }

            if (s < min) {
                min = s;
                rx = deq[d].x;
                ry = deq[d].y;
                result = 1 - (float(min) / max);
            }
            if (s == 0) {
                break;
            }
        }
    }

    cout << "Result: " << result << endl;
    if (result) {
        cout << "Found at [" << rx << "," << ry << "]" << endl;
        rectangle(haystack, Point(rx, ry), Point(rx + needle.cols, ry + needle.rows), Scalar::all(0), 2, 8, 0);
        imshow("Result", haystack);
        waitKey(0);
    }

    return 0;
}
