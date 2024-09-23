/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>
#include <map>

namespace
{
struct tree_node
{
   cv::Mat data;
   bool has_childrens;
   std::vector<tree_node> childrens;
   int x_l, x_r, y_b, y_t;

   tree_node(cv::Mat data, int x_l = 0, int x_r = 0, int y_b = 0, int y_t = 0) : 
            data(data), has_childrens(false), x_l(x_l), x_r(x_r), y_b(y_b), y_t(y_t) {}
};

void split_image(cv::Mat image, double stddev, tree_node* node)
{
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    node->has_childrens = true;

    const auto width = image.cols;
    const auto height = image.rows;

    cv::Mat up_left = image(cv::Range(0, height / 2), cv::Range(0, width / 2));
    cv::Mat down_left = image(cv::Range(0, height / 2), cv::Range(width / 2, width));
    cv::Mat up_right = image(cv::Range(height / 2, height), cv::Range(0, width / 2));
    cv::Mat down_right = image(cv::Range(height / 2, height), cv::Range(width / 2, width));

    tree_node up_left_child(up_left, 0, height / 2, 0, width / 2);
    tree_node down_left_child(down_left, 0, height / 2, width / 2, width);
    tree_node up_right_child(up_right, height / 2, height, 0, width / 2);
    tree_node down_right_child(down_right, height / 2, height, width / 2, width);

    node->childrens.push_back(up_left_child);
    node->childrens.push_back(down_left_child);
    node->childrens.push_back(up_right_child);
    node->childrens.push_back(down_right_child);

    split_image(up_left, stddev, &up_left_child);
    split_image(down_left, stddev, &down_left_child);
    split_image(up_right, stddev, &up_right_child);
    split_image(down_right, stddev, &down_right_child);
}

void merge_nodes(std::vector<tree_node*>& lists)
{
    cv::Mat mean1, mean;
    cv::Mat dev1;

    for (int i = 0; i < lists.size(); ++i)
    {
        cv::meanStdDev(lists[i]->data, mean1, dev1);
        if (i == 0)
        {
            mean = mean1;
            continue;
        }
        mean += mean1;
    }

    mean = mean/lists.size();

    for (int i = 0; i < lists.size(); ++i)
    {
        lists[i]->data.setTo(mean);
    }
}

void create_vector_for_childrens(tree_node* node, std::vector<tree_node*>& lists)
{
    if (!node->has_childrens)
    {
        lists.push_back(node);
        return;
    }

    for (auto& child : node->childrens)
    {
        create_vector_for_childrens(&child, lists);
    }
}

bool is_neibours(tree_node* node1, tree_node* node2)
{
    return node1->x_l == node2->x_r || node1->y_b == node2->y_t || node1->x_r == node2->x_l || node1->y_t == node2->y_b;
}

bool can_be_merge(tree_node* node1, tree_node* node2, double stddev)
{
    cv::Mat mean1, mean2;
    cv::Mat dev1, dev2;

    cv::meanStdDev(node1->data, mean1, dev1);
    cv::meanStdDev(node2->data, mean2, dev2);

    return dev1.at<double>(0) <= stddev && dev2.at<double>(0) <= stddev && is_neibours(node1, node2);
}

void merge_image_parts(std::vector<tree_node*>& lists, double stddev)
{
    std::multimap<int, tree_node*> group_node;
    int count = 0;

    for(int i = 0; i < lists.size(); ++i)
    {
        if (i == 0)
        {
            group_node.insert(std::pair<int, tree_node*>{++count, lists[i]});
            continue;
        }

        bool added_flag = false;

        for (int j = 1; j <= count; ++j)
        {
            auto range = group_node.equal_range(j);
            for (auto iter = range.first; iter != range.second; ++iter)
            {
                if (can_be_merge(iter->second, lists[i], stddev))
                {
                    group_node.insert(std::pair<int, tree_node*>{j, lists[i]});
                    added_flag = true;
                    break;
                }
            }
            if (added_flag)
                break;
        }
        if (!added_flag)
            group_node.insert(std::pair<int, tree_node*>{++count, lists[i]});
    }

    std::vector<tree_node*> nodes_vector;
    for (int i = 1; i <= count; ++i)
    {
        auto range = group_node.equal_range(i);
        for (auto iter = range.first; iter != range.second; ++iter)
        {
            nodes_vector.push_back(iter->second);
        }
        merge_nodes(nodes_vector);
        nodes_vector.clear();
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    cv::Mat res = image;
    tree_node node(res);

    std::vector<tree_node*> lists;

    split_image(res, stddev, &node);
    create_vector_for_childrens(&node, lists);
    merge_image_parts(lists, stddev);

    return res;
}
} // namespace cvlib
