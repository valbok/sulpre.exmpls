/**
 * @author VaL Doroshchuk <valbok@gmail.com>
 * @created 26 Sep 2015
 */

/**
 * Command line application that can read a single image or a directory of images and detect all faces + eyes in the image. 
 * Draws rectangle around each face and eye and writes the output to a new file or directory. 
 */

#include <string>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/**
 * Handler to detect faces and eyes.
 */
class Detector {
public:

    /**
     * @params Classifiers
     */
    Detector(CascadeClassifier* faceCascade,
        CascadeClassifier* eyesCascade) : mFaceCascade(faceCascade), mEyesCascade(eyesCascade) {}
    ~Detector() {
        delete mFaceCascade;
        delete mEyesCascade;
    }

    /**
     * Detects faces and eyes based on provided cascades.
     * @param Image
     * @param Output filename to store result image. If not provided will show a dialog.
     * @return true If found.
     */
    bool detect(Mat& image, const string& output) {
        bool result = false;
        std::vector<Rect> faces;
        Mat gray;

        cvtColor(image, gray, CV_BGR2GRAY);
        equalizeHist(gray, gray);

        mFaceCascade->detectMultiScale(gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces.size(); ++i) {
            result = true;
            Point pt1(faces[i].x, faces[i].y);
            Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
            rectangle(image, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);

            Mat faceROI = gray(faces[i]);
            std::vector<Rect> eyes;
            mEyesCascade->detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30));
            for (size_t j = 0; j < eyes.size(); ++j) {
                Point pt1(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
                Point pt2((faces[i].x + eyes[j].x + eyes[j].height), (faces[i].y + eyes[j].y + eyes[j].width));
                rectangle(image, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
            }
        }
        if (output.empty()) {
            imshow("Facedetect", image);
            waitKey(0);
            return result;
        }

        return result && imwrite(output, image);
    }

private:

    /**
     * Classifiers
     */
    CascadeClassifier* mFaceCascade;
    CascadeClassifier* mEyesCascade;
};

/**
 * Factory to create a detector.
 * @return 0 If something bad occured.
 */
Detector* createDetector(const char* faceCascadeFilename, const char* eyesCascadeFilename) {
    CascadeClassifier* faceCascade = new CascadeClassifier;
    CascadeClassifier* eyesCascade = new CascadeClassifier;
    if (!faceCascade->load(faceCascadeFilename) || !eyesCascade->load(eyesCascadeFilename)) {
        delete faceCascade;
        delete eyesCascade;

        return 0;
    }

    return new Detector(faceCascade, eyesCascade);
}

/**
 * Handler to process submitted path.
 */
class Reader {
public:

    /**
     * @param Detector
     * @param Output directory or filename to store the result.
     */
    Reader(Detector& detector, const string& output) : mDetector(detector), mOutput(output) {
    }

    /**
     * Reads path and decides how to proccess it based on either dir or file.
     * @return true If found a face.
     */
    bool read(const string& path) {
        if (is_dir(path)) {
            return readDir(path);
        }

        return detect(path, mOutput);
    }

private:

    /**
     * Detects faces using path to file.
     * @param Path to image
     * @param Output filename to store the result.
     */
    bool detect(const string& path, const string& output) {
        Mat image = imread(path.c_str(), CV_LOAD_IMAGE_COLOR);
        bool result = false;
        if (image.data) {
            try {
                result = mDetector.detect(image, output);
            } catch (Exception& e) {
                cerr << e.what() << endl;
            }
        }
        return result;
    }

    /**
     * Reads the dir by path.
     * @param true If found result.
     */
    bool readDir(const string& path) {
        bool result = false;
        DIR* dp = opendir(path.c_str());
        if (dp == 0) {
            return result;
        }

        struct dirent *dirp;
        while (dirp = readdir(dp)) {
            string name = dirp->d_name;
            if (name == "." || name == "..") {
                continue;
            }
            string filepath = path + "/" + name;
            if (is_dir(filepath)) {
                result = readDir(filepath);
            } else {
                string o = mOutput.empty() ? mOutput : mOutput + "/" + name;
                result = detect(filepath, o);
            }
        }

        closedir(dp);
        return result;
    }

    /**
     * Checks if path is dir.
     * @return true If yes.
     */
    static bool is_dir(const string& path) {
        struct stat buf;
        stat(path.c_str(), &buf);
        return S_ISDIR(buf.st_mode);
    }

    /**
     * Detector injection.
     */
    Detector& mDetector;

    /**
     * Output finename or dirname to store result.
     */
    const string& mOutput;
};

void showHelp(const char *appName) {
    cerr <<  "Usage: " << appName << " FILENAME-or-DIR [OUTPUT_FILENAME-or-DIR]\n";
}

int main(int argc, const char** argv) {
    string path;
    string output;

    if (argc > 1) {
        path = argv[1];
    }
    if (argc > 2) {
        output = argv[2];
    }

    if (path.empty()) {
        showHelp(argv[0]);
        return 1;
    }

    Detector* detector = createDetector("haarcascade_frontalface_alt.xml",
        "haarcascade_eye_tree_eyeglasses.xml");

    if (detector == 0) {
        cout << "Could not load cascade files." << endl;
        return 2;
    }

    Reader reader(*detector, output);
    if (!reader.read(path)) {
        cerr << "Could not find any faces in " << path << endl;
    }

    delete detector;
    return 0;
}