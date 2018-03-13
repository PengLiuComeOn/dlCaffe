/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"
#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";

double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;
int mode=0;
bool firstRosSpin=false;
using namespace dso;

void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}

void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}

void parseArgument(char* arg)
{
	int option = -1;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}

	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	if(1==sscanf(arg,"start=%d",&option))
	{
		start = option;
		printf("START AT %d!\n",start);
		return;
	}
	if(1==sscanf(arg,"end=%d",&option))
	{
		end = option;
		printf("END AT %d!\n",start);
		return;
	}

	if(1==sscanf(arg,"files=%s",buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}
    
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}

//#### 3.2 Commandline Options
//there are many command line options available, see `main_dso_pangolin.cpp`. some examples include
//- `mode=X`:
//-  `mode=0` use iff a photometric calibration exists (e.g. TUM monoVO dataset).
//-  `mode=1` use iff NO photometric calibration exists (e.g. ETH EuRoC MAV dataset).
//-  `mode=2` use iff images are not photometrically distorted (e.g. synthetic datasets).
//
//- `preset=X`
//- `preset=0`: default settings (2k pts etc.), not enforcing real-time execution
//- `preset=1`: default settings (2k pts etc.), enforcing 1x real-time execution
//- `preset=2`: fast settings (800 pts etc.), not enforcing real-time execution. WARNING: overwrites image resolution with 424 x 320.
//- `preset=3`: fast settings (800 pts etc.), enforcing 5x real-time execution. WARNING: overwrites image resolution with 424 x 320.
//
//- `nolog=1`: disable logging of eigenvalues etc. (good for performance)
//- `reverse=1`: play sequence in reverse
//- `nogui=1`: disable gui (good for performance)
//- `nomt=1`: single-threaded execution
//- `prefetch=1`: load into memory & rectify all images before running DSO.
//- `start=X`: start at frame X
//- `end=X`: end at frame X
//- `speed=X`: force execution at X times real-time speed (0 = not enforcing real-time)
//- `save=1`: save lots of images for video creation
//- `quiet=1`: disable most console output (good for performance)
//- `sampleoutput=1`: register a "SampleOutputWrapper", printing some sample output data to the commandline. meant as example.

int main( int argc, char** argv )
{
	//setlocale(LC_ALL, "");
    int Argc = 18;
    char* Argv[] = {
        "sampleoutput=1", "quiet=0", "preset=0", "noros=1","nolog=0", "reverse=0",
        "nogui=0", "nomt=0", "prefetch=0", "start=3000", "end=10000", "speed=0",
        "save=1", "mode=2", "rescale=1", "rec=0",
//        "files=/Users/conti-app-ma-033/Desktop/image_0",
//        "calib=/Users/conti-app-ma-033/Documents/Project/DSO-XCODE/sequence_11/cameraGM.txt",
//        "files=/Users/conti-app-ma-033/Desktop/ORBDSO/image_0",
//        "calib=/Users/conti-app-ma-033/Desktop/cameraiphone.txt",
        
        "files=/home/test/RoadDB/DSO_Code/sequence_14/images.zip",
        "calib=/home/test/RoadDB/DSO_Code/sequence_14/camera.txt",
        
//        "vignette=/Users/conti-app-ma-033/Documents/Project/DSO-XCODE/sequence_11/vignette.png",
//        "gamma=/Users/conti-app-ma-033/Documents/Project/DSO-XCODE/sequence_11/pcalib.txt"
    };
    
    std::string strPose = "/Users/conti-app-ma-033/Desktop/all_slam_pose.txt";
    
	for(int i=0; i<Argc;i++)
		parseArgument(Argv[i]);
	
	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);
    //读取文件并定义undistort对象：在其构造函数中对相机内参进行矫正，读取行光标定参数，读取每张图像的曝光时间；
	ImageFolderReader* reader = new ImageFolderReader(source, calib, gammaCalib, vignette);  
	reader->setGlobalCalibration();

	if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)  //setting_photometricCalibration是大于0，这里暂时不进行光度标定；
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}

	int lstart=start;		//开始处理的图像帧，在前面参数中设置；
	int lend = end;			//结束处理的图像帧，在前面参数中设置；
	int linc = 1;   // linc：每次间隔多张图片处理　
	if(reverse)     // reverse ==0, 如果它等于１，则逆序遍历；
	{
		printf("REVERSE!!!!");
		lstart=end-1;
		if(lstart >= reader->getNumImages())
			lstart = reader->getNumImages()-1;
		lend = start;
		linc = -1;
	}
	
    // 管理帧和点的类，在构造函数中创建日志文件和初始化一些类对象，创建mapping类
	FullSystem* fullSystem = new FullSystem();    
    
    fullSystem->loadPosePrecalc(strPose);   //这个strPose对应的文件是无效的，所以这一步无用，会直接返回；
	fullSystem->setGammaFunction(reader->getPhotometricGamma());  //这步也直接返回，不会去设置gamma函数(因为传入参数没有gamma矫正文件);
	fullSystem->linearizeOperation = (playbackSpeed==0); //playbackSpeed==0,　所以设置linearizeOperation==1;
    
    lstart = fullSystem->firstFrameId;    // firstFrameId == 0;
    
    std::cout << "firstFrameId: " << fullSystem->firstFrameId << std::endl;
  
    IOWrap::PangolinDSOViewer* viewer = 0;    // 图像显示类
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;           //存储要处理图片的序号
        std::vector<double> timesToPlayAt;    // 时间容器
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);   // tsPrev == tsThis == 1.46123e+09 ????
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);   //??????这里playbackSpeed为０，可以作为除数？??输出结果里面该容器的值都为inf;
            }
       
	 
        }

	std::vector<ImageAndExposure*> preloadedImages;  //包含图像辐照度和曝光时间的图像类
	if(preload)   //用于提前加载所有图片，但是这里没用
	{
		printf("LOADING ALL IMAGES!\n");
		for(int ii=0;ii<(int)idsToPlay.size(); ii++)
		{
			int i = idsToPlay[ii];
			preloadedImages.push_back(reader->getImage(i));
		}
	}

	struct timeval tv_start;
	gettimeofday(&tv_start, NULL);                   //用于处理时间的函数
	clock_t started = clock();
	double sInitializerOffset=0;

	for(int ii=0;ii<(int)idsToPlay.size(); ii++)
	{
		if(!fullSystem->initialized)	// if not initialized: reset start time.
		{
			gettimeofday(&tv_start, NULL);
			started = clock();
			sInitializerOffset = timesToPlayAt[ii];
		}

		int i = idsToPlay[ii];

		ImageAndExposure* img;
    	if(preload)
    		img = preloadedImages[ii];
    	else
    		img = reader->getImage(i);         // 读取图像,在这里对读取的图像进行光度校正；

		bool skipFrame=false;
		if(playbackSpeed!=0)                  //playbackSpeed的作用是什么？？？
		{
			struct timeval tv_now; gettimeofday(&tv_now, NULL);
			double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

			if(sSinceStart < timesToPlayAt[ii])
				usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
			else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
			{
				printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
				skipFrame=true;
			}
		}

		if(!skipFrame) fullSystem->addActiveFrame(img, i);       //开启整个SLAM的初始化，定位和建图；

		delete img;

		if(fullSystem->initFailed || setting_fullResetRequested)
		{
			if(ii < 250 || setting_fullResetRequested)
			{
				printf("RESETTING!\n");

				std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
				delete fullSystem;

				for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

				fullSystem = new FullSystem();
				fullSystem->setGammaFunction(reader->getPhotometricGamma());
				fullSystem->linearizeOperation = (playbackSpeed==0);


				fullSystem->outputWrapper = wraps;

				setting_fullResetRequested=false;
			}
		}

		if(fullSystem->isLost)
		{
            printf("LOST!!\n");
            break;
		}

	}
	fullSystem->blockUntilMappingIsFinished();
	clock_t ended = clock();
	struct timeval tv_end;
	gettimeofday(&tv_end, NULL);

	fullSystem->printResult("result.txt");

	int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
	double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
	double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
	double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	printf("\n======================"
			"\n%d Frames (%.1f fps)"
			"\n%.2fms per frame (single core); "
			"\n%.2fms per frame (multi core); "
			"\n%.3fx (single core); "
			"\n%.3fx (multi core); "
			"\n======================\n\n",
			numFramesProcessed, numFramesProcessed/numSecondsProcessed,
			MilliSecondsTakenSingle/numFramesProcessed,
			MilliSecondsTakenMT / (float)numFramesProcessed,
			1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
			1000 / (MilliSecondsTakenMT / numSecondsProcessed));
    //fullSystem->printFrameLifetimes();
	if(setting_logStuff)
	{
		std::ofstream tmlog;
		tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
		tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
			  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
		tmlog.flush();
		tmlog.close();
	}
        

    });


    if(viewer != 0)
        viewer->run();

    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;
	printf("DELETE READER!\n");
	delete reader;
	printf("EXIT NOW!\n");
	return 0;
}
