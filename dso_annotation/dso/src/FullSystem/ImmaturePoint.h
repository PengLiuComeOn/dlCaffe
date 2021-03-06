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


#pragma once

 
#include "util/NumType.h"
 
#include "FullSystem/HessianBlocks.h"
namespace dso
{


struct ImmaturePointTemporaryResidual
{
public:
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	FrameHessian* target;
};


enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good
	IPS_OOB,					// OOB: end tracking & marginalize!
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	IPS_BADCONDITION,			// not traced because of bad condition.
	IPS_UNINITIALIZED};			// not even traced once.


class ImmaturePoint
{
public:
	// static values
	float color[MAX_RES_PER_POINT];   //存储对应immature点周围8个像素的像素值
	float weights[MAX_RES_PER_POINT];  //存储immature点周围8个像素的权重，用于累计计算重投影误差；
	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	float u,v;
	FrameHessian* host;
	int idxInImmaturePoints;  //FrameHession里包含immature点的容器中对应于该immature的索引，以便对该点的操作或删除；

	float quality;

	float my_type;

	float idepth_min;
	float idepth_max;
	ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib);
	~ImmaturePoint();

	ImmaturePointStatus traceOn(FrameHessian* frame, Mat33f hostToFrame_KRKi, Vec3f hostToFrame_Kt, Vec2f hostToFrame_affine, CalibHessian* HCalib, bool debugPrint=false);

	ImmaturePointStatus lastTraceStatus;
	Vec2f lastTraceUV;
	float lastTracePixelInterval;   //最大和最小对极线上的距离

	float idepth_GT;

	double linearizeResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float &Hdd, float &bd,
			float idepth);
	float getdPixdd(
			CalibHessian *  HCalib,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	float calcResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

private:
};

}

