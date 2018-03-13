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

namespace dso
{
struct RawResidualJacobian
{
	// ================== new structure: save independently =============.
	EIGEN_ALIGN16 VecNRf resF;             //投影像素值的差res*hw： 有8个元素对应每个像素周围8个像素的残差

	// the two rows of d[x,y]/d[xi].
	EIGEN_ALIGN16 Vec6f Jpdxi[2];			// 2x6   diff pose  像素坐标对pose求导

	// the two rows of d[x,y]/d[C].
	EIGEN_ALIGN16 VecCf Jpdc[2];			// 2x4   diff camera (K) 像素坐标对K求导

	// the two rows of d[x,y]/d[idepth].
	EIGEN_ALIGN16 Vec2f Jpdd;				// 2x1   diff depth   像素坐标对depth求导

	// the two columns of d[r]/d[x,y].
	EIGEN_ALIGN16 VecNRf JIdx[2];			// 9x2   diff (u,v)   像素值对坐标求导

	// = the two columns of d[r] / d[ab]
	EIGEN_ALIGN16 VecNRf JabF[2];			// 9x2   diff a b 像素值对a,b求导 （原图像的像素值 (gray - b)*hw， hw）


	// = JIdx^T * JIdx (inner product). Only as a shorthand.
	EIGEN_ALIGN16 Mat22f JIdx2;				// 2x2    [dx*dx, dx*dy; dx*dy, dy*dy] (每个元素都对对应点周围8个像素做累加)
	// = Jab^T * JIdx (inner product). Only as a shorthand.
	EIGEN_ALIGN16 Mat22f JabJIdx;			// 2x2    [像素对a求导*dx,像素对a求导*dy; 像素对b求导*dx,像素对b求导*dy] （每个元素都对对应点周围8个像素做累加）
	// = Jab^T * Jab (inner product). Only as a shorthand.
	EIGEN_ALIGN16 Mat22f Jab2;			// 2x2  [像素对a求导*像素对a求导，像素对a求导*像素对b求导 ;像素对a求导*像素对b求导, 像素对b求导*像素对b求导] （每个元素都对对应点周围8个像素做累加）

};
}

