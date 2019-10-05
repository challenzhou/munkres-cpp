// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tracking_manager.hpp"
#include <omp.h>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

namespace tracker
{


void TrackingManager::matchTrackDet(cv::Mat& weights, cv::Mat& matches)
{
  int origin_rows = weights.rows, origin_cols = weights.cols;
  int size_squal = (origin_rows>origin_cols)?origin_rows:origin_cols;

  /*extend weight to be squal size*/
  cv::Mat correlations = cv::Mat::zeros(size_squal, size_squal, CV_32F);
  cv::Mat clip = correlations(cv::Rect(0, 0, origin_cols, origin_rows));
  weights.copyTo(clip);

  /*initial matches*/
  cv::Mat row_mask = cv::Mat(1, size_squal, CV_32SC1, cv::Scalar(-1)); 
  cv::Mat col_mask = cv::Mat(1, size_squal, CV_32SC1, cv::Scalar(-1)); 

  /*max weights*/
  cv::Mat row_weight = cv::Mat::zeros(1, size_squal, CV_32F); 
  cv::Mat col_weight = cv::Mat::zeros(1, size_squal, CV_32F); 

  /*initial visit mask for each round of hungarian search*/
  cv::Mat row_visit = cv::Mat::zeros(1, size_squal, CV_8UC1); 
  cv::Mat col_visit = cv::Mat::zeros(1, size_squal, CV_8UC1); 

  /*initial visit mask for each round of hungarian search*/
  cv::Mat col_gap= cv::Mat(1, size_squal, CV_32F, cv::Scalar(INFINITY)); 

  /*initialize row_weight, col_weight*/
  for (int i=0; i<size_squal; i++) {
    int row_maxId = -1, col_maxId = -1;
    float row_max = 0.0f, col_max = 0.0f;

    for (int j=0;j<size_squal; j++) {
      float row_value = correlations.at<float>(i, j);
      if (row_max <= row_value)
      {
        row_max = row_value;
        row_maxId = j;
      }

      float col_value = correlations.at<float>(j, i);
      if (col_max <= col_value)
        col_max = col_value;
        col_maxId = j;
    }

   // row_mask.at<uint8_t>(i) = row_maxId;
    row_weight.at<float>(i) = row_max;

   // col_mask.at<int32_t>(i) = col_maxId;
  }

  /*search from tracker to detection*/
  for (int i=0; i<size_squal; i++) {
    /*reinitialize min weight gap for each search*/
    col_gap = cv::Scalar(INFINITY);
 
    while (true) {
        /*reinitialize visit flags*/
        row_visit = 0; 
        col_visit = 0; 
        bool ret = searchMatch(i, row_visit, row_weight, col_visit,col_weight,
                               col_mask,col_gap,correlations); 
        if (!ret) {
          int min_idx[2];
          cv::minMaxIdx(col_gap, NULL, NULL, min_idx);
          float min_gap = col_gap.at<float>(min_idx[0], min_idx[1]);

          for (int j = 0; j<size_squal; j++) {
            if (col_visit.at<uint8_t>(j) == 1)
              col_weight.at<float>(j) += min_gap;
            else
              col_gap.at<float>(j) -= min_gap;

            if (row_visit.at<uint8_t>(j) == 1)
              row_weight.at<float>(j) -= min_gap;
          }

       } else {

          break;
       } 

    }

  }

  cv::Mat res = col_mask(cv::Rect(0,0,matches.cols, matches.rows));
  res.copyTo(matches);

}

/*search for each row item, to match col item*/
bool TrackingManager::searchMatch(
  int srcId,
  cv::Mat& srcVisit,
  cv::Mat& srcCorr,
  cv::Mat& tgtVisit,
  cv::Mat& tgtCorr,
  cv::Mat& tgtMatch,
  cv::Mat& weightDelta,
  cv::Mat& correlations)
{
  int tgt_size = tgtCorr.cols;

  srcVisit.at<uint8_t>(srcId) = 1;

  float srcCorrValue = srcCorr.at<float>(srcId);
  for (int i=0; i<tgt_size; i++)
  {
    if (tgtVisit.at<uint8_t>(i) == 1)
      continue;

    float gap = srcCorrValue + tgtCorr.at<float>(i) - correlations.at<float>(srcId,i);
    if (abs(gap <= 1e-04)) gap = 0.0f;

    if (gap == 0.0f)
    {
      tgtVisit.at<uint8_t>(i) = 1;
      int tgtSrcIdx = tgtMatch.at<int32_t>(i);
      if ((tgtSrcIdx == -1) || 
           searchMatch(tgtSrcIdx, srcVisit, srcCorr, tgtVisit, tgtCorr,
                       tgtMatch, weightDelta, correlations) )
      {
        tgtMatch.at<int32_t>(i) = srcId;
        //srcMatch.at<int32_t>(srcId) = i;

        return true;

      }

    } else {
        weightDelta.at<float>(i) = std::min(gap, weightDelta.at<float>(i));
    }

  }

  return false;
}



}  // namespace tracker
