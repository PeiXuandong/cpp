#include <vector>
#include <array>
#include <functional>
#include <cmath>
#include <map>
#include <numbers> 
#define _USE_MATH_DEFINES  // 启用数学常量（MSVC 需要）
#include <cmath>           // 包含数学函数

#ifndef M_PI               // 如果 M_PI 未定义，手动定义
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using UpOrBelowType = vector<vector<array<array<bool, 2>, 2>>>;//为全局的up_or_blew定义类型


class Element {
public:
    // 单元坐标信息
    struct Vertex {
        double x;
        double y;
    };

    array<array<bool, 2>, 2> up_or_below;  // 顶点位置关系
    int partition;                         // 分区类型
    bool is_interface;                     // 是否为界面单元

    // 构造函数
    Element(int i, int j, int N, const function<bool(double, double)>& on_either_part) {
        const double h = 1.0 / N;
        const double x = (j - 1) * h;
        const double y = (i - 1) * h;

        // 计算四个顶点位置关系
        up_or_below[0][0] = on_either_part(x, y);         // 左下
        up_or_below[0][1] = on_either_part(x + h, y);      // 右下
        up_or_below[1][0] = on_either_part(x, y + h);      // 左上
        up_or_below[1][1] = on_either_part(x + h, y + h); // 右上

        // 判断是否为界面单元
        const int sum = (up_or_below[0][0] ? 1 : 0) +
            (up_or_below[0][1] ? 1 : 0) +
            (up_or_below[1][0] ? 1 : 0) +
            (up_or_below[1][1] ? 1 : 0);

        is_interface = (sum % 4 != 0);

        // 确定分区类型
        if (sum == 0) {
            partition = 0;
        }
        else if (sum == 4) {
            partition = 1;
        }
        else {
            partition = -1;
        }
    }
};

// 网格处理函数
struct InterfaceResult {
    vector<vector<bool>> inter;
    vector<vector<Element>> elements;
    vector<vector<int>> partition;
    vector<pair<int, int>> echVerNodeFreq;
    vector<array<int, 9>> echFaceNodeFreq;
    UpOrBelowType global_up_or_blew;
};

void intersection_ext(
    int i, int j,
    const array<array<bool, 2>, 2>& C,              // 2x2 矩阵，每个元素表示角点是否在 part0 或 part1
    int N,
    const function<double(double)>& intfacex,       // 函数对象，计算 x 坐标
    const function<double(double)>& intfacey,       // 函数对象，计算 y 坐标
    vector<double>& x,                              // 输出 x 坐标
    vector<double>& y                               // 输出 y 坐标
) {
    x.clear();  // 清空输出
    y.clear();

    double h = 1.0 / N;  // 网格步长
    double x0 = (j)*h;  // 左下角 x 坐标
    double y0 = (i)*h;  // 左下角 y 坐标

    // 计算四个边的和
    int left = C[0][0] + C[1][0];
    int bottom = C[0][0] + C[0][1];
    int right = C[0][1] + C[1][1];
    int top = C[1][0] + C[1][1];

    // 检查是否在边界上，并计算交点
    if (bottom == 1) {
        x.push_back(intfacey(y0));  // y0 对应的 x 坐标
        y.push_back(y0);
    }
    if (right == 1) {
        x.push_back(x0 + h);
        y.push_back(intfacex(x0 + h));  // x0 + h 对应的 y 坐标
    }
    if (top == 1) {
        x.push_back(intfacey(y0 + h));  // y0 + h 对应的 x 坐标
        y.push_back(y0 + h);
    }
    if (left == 1) {
        x.push_back(x0);
        y.push_back(intfacex(x0));  // x0 对应的 y 坐标
    }
}

InterfaceResult interect_ext(const function<bool(double, double)>& on_either_part, int N) {
    InterfaceResult result;
    const int grid_size = N + 1;

    // 初始化数据结构
    result.inter = vector<vector<bool>>(N, vector<bool>(N, false));
    result.elements = vector<vector<Element>>(N, vector<Element>(N, Element(0, 0, 1, on_either_part)));
    result.partition = vector<vector<int>>(N, vector<int>(N, 1));
    result.global_up_or_blew = UpOrBelowType(N, vector<array<array<bool, 2>, 2>>(N));
    map<int, int> enrich_node_vertex;
    vector<array<int, 9>> enrich_node_face((N + 1) * (N + 1), { 0 });

    // 第一遍扫描：处理单元基本属性
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Element elem(i + 1, j + 1, N, on_either_part);
            result.elements[i][j] = elem;
            result.inter[i][j] = elem.is_interface;
            result.partition[i][j] = elem.partition;
            result.global_up_or_blew[i][j] = elem.up_or_below;

            // 统计需要增强的顶点
            if (elem.partition == -1) {
                const int base_id = i * grid_size + j;
                const array<int, 4> vertexIds = {
                    base_id,
                    base_id + 1,
                    base_id + grid_size,
                    base_id + grid_size + 1
                };

                for (int id : vertexIds) {
                    enrich_node_vertex[id] = enrich_node_vertex[id] + 1;
                }
            }
        }
    }

    // 生成顶点频率表
    vector<pair<int, int>> echVerNodeFreq;
    for (const auto& entry : enrich_node_vertex) {
        const int& id = entry.first;
        const int& count = entry.second;
        result.echVerNodeFreq.emplace_back(id + 1, count); // MATLAB索引转换
    }

    // 第二遍扫描：处理面增强信息
    int num = 0;
    for (const auto& entry : enrich_node_vertex) {
        const int& id = entry.first;
        const int& count = entry.second;
        const int matlab_id = id; // 转换为MATLAB索引
        const int i = matlab_id / grid_size + 1;
        const int j = matlab_id % grid_size + 1;
        num++;

        // 四个方向的邻接面处理
        auto update_face = [&](int face_i, int face_j, int pos1, int pos2) {
            if (face_i >= 0 && face_j >= 0 && face_i < N && face_j < N) {
                const int face_id = face_i * N + face_j;
                enrich_node_face[face_id][pos1] = matlab_id + 1;
                enrich_node_face[face_id][pos2] = num;
            }
            };

        if (i > 1 && j > 1) update_face(i - 2, j - 2, 3, 7); // 左下
        if (i > 1 && j <= N) update_face(i - 2, j - 1, 1, 5); // 右下
        if (i <= N && j > 1) update_face(i - 1, j - 2, 2, 6); // 左上
        if (i <= N && j <= N) update_face(i - 1, j - 1, 0, 4); // 右上
    }

    // 生成最终面频率表
    for (size_t face_id = 0; face_id < enrich_node_face.size(); ++face_id) {
        auto& face = enrich_node_face[face_id];
        face[8] = 0; // 第9列存储总和
        int sum = 0;
        for (int k = 0; k < 8; ++k)
        {
            sum = sum + face[k];

        }

        for (int k = 0; k < 8; ++k)
        {
            face[8 - k] = face[8 - k - 1];
        }
        face[0] = face_id + 1; // MATLAB索引转换

        if (sum > 0) {
            array<int, 9> record;
            copy(face.begin(), face.end(), record.begin());
            result.echFaceNodeFreq.push_back(record);
        }
    }

    return result;

}
Eigen::VectorXd calc_kappa(
    const Eigen::MatrixXd& gps,
    double (*intface)(double)) {  // 或用 std::function<double(double)>

    const int n = gps.rows();
    Eigen::VectorXd res(n);  // 结果 VectorXd

    for (int i = 0; i < n; ++i) {
        double x = gps(i, 0);          // gps 的第一列
        double y = intface(x);         // 计算 y = intface(x)
        res(i) = (y > gps(i, 1)) ? 1.0 : 0.0;  // 比较结果转为 double
    }

    return res;
}

double calcJacobiDetValues(int ElemID, int N) {
    double h = 1.0 / static_cast<double>(N);  // 转换为浮点数除法
    double jac = h * h;

    // 计算元素左下角坐标（注释掉的MATLAB代码）
    // int offJ = (ElemID - 1) % N;  // C++的模运算
    // int offI = (ElemID - 1) / N;   // 整数除法自动floor
    // double x0 = offJ * h;
    // double y0 = offI * h;

    return jac;
}

// 使用示例

