#pragma once
#include <vector>
#include <array>
#include <algorithm>

#include <Eigen/Dense>

using namespace Eigen;

using namespace std;

/**
 * @brief 模拟 MATLAB 的 ismember 函数，检查 elemID 是否存在于 echFaceNodeFreq 的指定列
 *
 * @param elemID 要查找的 ID
 * @param echFaceNodeFreq 数据矩阵（vector<array<int, 9>>）
 * @param column_index 要检查的列索引（从 0 开始）
 * @param pos 输出参数，存储找到的位置（从 1 开始计数，未找到返回 -1）
 * @return true 如果找到，false 如果未找到
 */

bool Ismember_single(
    int elemID,
    const std::vector<std::array<int, 9>>& echFaceNodeFreq,
    int column_index,
    int& pos
) {
    // 检查列索引是否有效
    if (column_index < 0 || column_index >= static_cast<int>(echFaceNodeFreq.empty() ? 0 : echFaceNodeFreq[0].size())) {
        pos = -1;
        return false; // 列索引无效
    }

    // 遍历查找指定列等于 elemID 的行
    auto it = std::find_if(echFaceNodeFreq.begin(), echFaceNodeFreq.end(),
        [elemID, column_index](const std::array<int, 9>& row) { return row[column_index] == elemID; });

    if (it != echFaceNodeFreq.end()) {
        pos = static_cast<int>(std::distance(echFaceNodeFreq.begin(), it)); // 从开始计数
        return true;
    }
    else {
        pos = -1; // 未找到
        return false;
    }
}

std::vector<size_t> Find(const std::vector<int>& vec) {
    vector<size_t> indices;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] != 0) {  // 检查是否非零
            indices.push_back(i);
        }
    }
    return indices;
}

void ismember(const std::vector<int>& faceIds, const std::vector<int>& IntElem,
    std::vector<bool>& flag, std::vector<int>& EchFuncId) {
    flag.resize(faceIds.size(), false);
    EchFuncId.resize(faceIds.size(), 0);

    for (size_t i = 0; i < faceIds.size(); ++i) {
        auto it = std::find(IntElem.begin(), IntElem.end(), faceIds[i]);
        if (it != IntElem.end()) {
            flag[i] = true;
            EchFuncId[i] = static_cast<int>(std::distance(IntElem.begin(), it)) + 1;
        }
    }
}

pair<bool, int> ismember_single(int fid, const vector<int>& echLocalDPUMap) {
    // 遍历 echLocalDPUMap 查找 fid
    for (size_t i = 0; i < echLocalDPUMap.size(); ++i) {  // 使用 size_t 更安全
        if (echLocalDPUMap[i] == fid) {  // 直接使用 operator[]
            return { true, static_cast<int>(i) };  // 找到，返回 true 和索引
        }
    }

    return { false, -1 };  // 未找到，返回 false 和 -1
}


VectorXd concatenateVectors(const VectorXd& valuelist, const MatrixXd& gpsValues, const std::vector<int>& posF)
{
    // 1. 提取 gpsValues 的指定列
    MatrixXd selectedCols(gpsValues.rows(), posF.size());
    for (size_t i = 0; i < posF.size(); ++i)
    {
        selectedCols.col(i) = gpsValues.col(posF[i]);  // 
    }
    // 2. 将 selectedCols 展开成一维向量（按列优先）
    VectorXd flatSelected = Map<const VectorXd>(selectedCols.data(), selectedCols.size());

    // 3. 拼接到 valuelist 末尾（使用 std::vector）
    std::vector<double> concatenated;
    concatenated.insert(concatenated.end(), valuelist.data(), valuelist.data() + valuelist.size());
    concatenated.insert(concatenated.end(), flatSelected.data(), flatSelected.data() + flatSelected.size());

    return VectorXd::Map(concatenated.data(), concatenated.size());
}


