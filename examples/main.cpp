/*
 *   Copyright (c) 2007 John Weaver
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

/*
 * Some example code.
 *
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <time.h>

#include "munkres.h"
#include "tracking_manager.hpp"
#include "adapters/boostmatrixadapter.h"

unsigned long GetTickCount()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
}

int
main(int argc, char *argv[]) {
	int nrows = 2;
	int ncols = 2;
	
	if ( argc == 3 ) {
		nrows = atoi(argv[1]);
		ncols = atoi(argv[2]);
	}

	Matrix<float> matrix(nrows, ncols);
	cv::Mat weights(nrows, ncols, CV_32F);
	cv::Mat match_tm = cv::Mat::zeros(1, ncols, CV_32F);
	
	srandom(time(nullptr)); // Seed random number generator.

  matrix(0,0) = 10;
  matrix(0,1) = 2;
  matrix(1,0) = 6;
  matrix(1,1) = 2;

  weights.at<float>(0, 0) = 0.1f;
  weights.at<float>(0, 1) = 0.5f;
  weights.at<float>(1, 0) = 0.16666667f;
  weights.at<float>(1, 1) = 0.5f;

#if 0
	// Initialize matrix with random values.
	for ( int row = 0 ; row < nrows ; row++ ) {
		for ( int col = 0 ; col < ncols ; col++ ) {
			float value = (float)(random()%10 + 1);
			matrix(row,col) = value;
      weights.at<float>(row, col) = 1.0f/value;
		}
	}
#endif
	// Display begin matrix state.
	for ( int row = 0 ; row < nrows ; row++ ) {
		for ( int col = 0 ; col < ncols ; col++ ) {
			std::cout.width(2);
			std::cout << matrix(row,col) << ",";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

  


	// Apply Munkres algorithm to matrix.
	Munkres<float> m;
	//Time measurment
	uint64_t e1 = GetTickCount();
	m.solve(matrix);
	uint64_t e2 = GetTickCount();
	double t1 = (e2 - e1);
	std::cout <<  "\nMunkres time cost:  " << t1<< "milli seconds" << std::endl;

  std::cout << "\nTM weights:" << weights << "\n" << std::endl;
  tracker::TrackingManager tm;
  e1 = GetTickCount();
  tm.matchTrackDet(weights, match_tm);
  e2 = GetTickCount();
	double t2 = (e2 - e1);
	std::cout <<  "\nMunkres-TM time cost:  " << t2<< "milli seconds" << std::endl;
  


	// Display solved matrix.
	for ( int row = 0 ; row < nrows ; row++ ) {
		for ( int col = 0 ; col < ncols ; col++ ) {
			std::cout.width(2);
			std::cout << matrix(row,col) << ",";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	
	for ( int row = 0 ; row < nrows ; row++ ) {
		int rowcount = 0;
		for ( int col = 0 ; col < ncols ; col++  ) {
			if ( matrix(row,col) == 0 )
				rowcount++;
		}
		if ( rowcount != 1 )
			std::cerr << "Row " << row << " has " << rowcount << " columns that have been matched." << std::endl;
	}

	for ( int col = 0 ; col < ncols ; col++ ) {
		int colcount = 0;
		for ( int row = 0 ; row < nrows ; row++ ) {
			if ( matrix(row,col) == 0 )
				colcount++;
		}
		if ( colcount != 1 )
			std::cerr << "Column " << col << " has " << colcount << " rows that have been matched." << std::endl;
	}


  std::cout << "\nTM match result:" << match_tm << "\n" << std::endl;
	return 0;
}
