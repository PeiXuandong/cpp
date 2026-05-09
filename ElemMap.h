#include <vector>
#include <cmath>
#include <stdexcept>
using Eigen::MatrixXd;
using namespace std;

// 二维坐标点结构体
struct Point2D {
    double x;
    double y;
    Point2D(double x = 0, double y = 0) : x(x), y(y) {}
};

class ElemMap {
private:
    int meshDensity;  // 网格密度 N
    int order;        // 多项式阶数 d 
    int n_nodex;      // x方向节点总数

    // 辅助方法：安全计算行列索引

public:
    // 构造函数
    vector<int> calcIndex(int elem_id) const {
        if (elem_id < 1 || elem_id > meshDensity * meshDensity)
            throw out_of_range("Element ID out of range");
        return { (elem_id - 1) / meshDensity + 1, (elem_id - 1) % meshDensity + 1 };
    }

    ElemMap(int N, int d) : meshDensity(N), order(d) {
        n_nodex = d * N + 1;  // 计算x方向节点数
    }

    // 通过行列号获取单元ID (i,j从0开始)
    int getElemId(int i, int j) const {
        if (i<0 || i>meshDensity - 1 || j<0 || j>meshDensity - 1)
            throw out_of_range("Invalid row/col index");
        return (i)*meshDensity + j + 1;
    }

    // 获取单元的相邻单元ID矩阵（3x3邻域）
    vector<vector<int>> getOneRingFaces(int elem_id) const {
        vector<int> ij = calcIndex(elem_id);
        int i = ij[0], j = ij[1];
        vector<vector<int>> faceIds(3, vector<int>(3, 0));

        for (int di = -1; di <= 1; ++di) {  // 行偏移
            for (int dj = -1; dj <= 1; ++dj) {  // 列偏移
                int ni = i + di;
                int nj = j + dj;
                if (ni >= 1 && ni <= meshDensity && nj >= 1 && nj <= meshDensity) {
                    faceIds[di + 1][dj + 1] = getElemId(ni - 1, nj - 1);
                }
                // 越界位置保持为0
            }
        }
        return faceIds;
    }

    // 局部坐标转全局坐标（批量处理）
    MatrixXd localToGlobal(int elem_id, const MatrixXd& localPoints) const {
        vector<int> ij = calcIndex(elem_id);
        int i = ij[0], j = ij[1];
        const double h = 1.0 / meshDensity;
        const double x0 = (j - 1) * h;
        const double y0 = (i - 1) * h;

        // 假设localPoints是Nx2的矩阵，每行是一个点的(x,y)坐标
        MatrixXd globalPoints(localPoints.rows(), 2);

        for (int k = 0; k < localPoints.rows(); ++k) {
            globalPoints(k, 0) = localPoints(k, 0) * h + x0;
            globalPoints(k, 1) = localPoints(k, 1) * h + y0;
        }
        return globalPoints;
    }

    MatrixXd LocalOriginCoord(const std::vector<int>& elem_ids) const {
        MatrixXd origins(elem_ids.size(), 2);

        double h = 1.0 / meshDensity;

        for (size_t k = 0; k < elem_ids.size(); ++k) {
            int elem_id = elem_ids[k];
            if (elem_id < 1 || elem_id > meshDensity * meshDensity)
                throw out_of_range("Element ID out of range");

            int j = (elem_id - 1) % meshDensity + 1;
            int i = (elem_id - 1) / meshDensity + 1;

            origins.row(k) << (j - 1) * h, (i - 1)* h;
        }

        return origins;
    }
    // 获取标准FEM基函数全局ID集合
    vector<int> getFEMBasisIds(int elem_id) const {
        vector<int> ij = calcIndex(elem_id);
        int i = ij[0], j = ij[1];
        const int base_i = order * (i - 1);
        const int base_j = order * (j - 1);
        const int nodesPerDim = order + 1;

        vector<int> ids;
        ids.reserve(nodesPerDim * nodesPerDim);

        // 改为列优先顺序：先固定列，遍历行
        for (int dj = 0; dj <= order; ++dj) {
            for (int di = 0; di <= order; ++di) {
                ids.push_back((base_i + di) * n_nodex + (base_j + dj + 1));
            }
        }
        return ids;
    }

    // 判断是否为边界单元
    bool isBoundary(int elem_id) const {
        vector<int> ij = calcIndex(elem_id);
        int i = ij[0], j = ij[1];
        return (i == 1 || i == meshDensity || j == 1 || j == meshDensity);
    }

    // 获取顶点邻域节点ID（优化版）
    vector<int> getVertexNeighbors(int elem_id) const {
        vector<int> ij = calcIndex(elem_id);
        int i = ij[0], j = ij[1];
        const int d = order;
        const int N = meshDensity;

        vector<int> nodeIds;
        nodeIds.reserve(9);  // 3x3邻域

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                int ni = (i - 2) * d + 1 + di * d;
                int nj = (j - 2) * d + 1 + dj * d;
                if (ni >= 1 && ni <= N * d + 1 && nj >= 1 && nj <= N * d + 1) {
                    nodeIds.push_back((ni - 1) * (N * d + 1) + nj);
                }
            }
        }
        return nodeIds;
    }


    void LocalOriginCoordPU(int elem_ids,
        int neighboor,
        double& x0,
        double& y0) const
    {

        // 计算网格间距 h
        double h = 1.0 / static_cast<double>(meshDensity);

        // 只计算第一个 elem_id 的结果（假设 neighboor 是固定的）
        int j = (elem_ids - 1) % meshDensity + (neighboor - 1) / 3 - 1;
        int i = (elem_ids - 1) / meshDensity + (neighboor - 1) % 3 - 1;

        x0 = (j - 1) * h;
        y0 = (i - 1) * h;
    }

    // 其他方法的实现...
    // [可根据需要添加更多方法实现]
};

