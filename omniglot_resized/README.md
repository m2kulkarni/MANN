
## HEADER

  

```c++

#include<depth_stereo.hpp>

```

  

## PUBLIC MEMBER FUNCTIONS:

  

```c++

depth_generator (const ros::NodeHandle& nh,
    const std::string& image_left_topic = "image1",
	const std::string& image_right_topic = "image2")

```

```c++

depth_generator (const ros::NodeHandle& nh,
	const std::map<std::string, float>& params,
	const std::string& image_left_topic = "image1",
	const std::string& image_right_topic = "image2")

```

```c++

void depth_calc (cv::Mat& depth = cv::Mat(),
	cv::Mat& disparity = cv::Mat())

```

  

## PROTECTED MEMBER FUNCTIONS

  

```c++

void left_update_callback(const sensor_msgs::ImageConstPtr& left)

```

```c++

void right_update_callback(const sensor_msgs::ImageConstPtr& right)

```

  

## PROTECTED ATTRIBUTES

  

```c++

int  num_disparities_  //the disparity search range. Used by cv::stereoBM(Default: 2*16)

```

```c++

int block_size_  //Blocks used by the algorithm. Used by cv::stereoBM(Default: 5)

```

```c++

int lambda_  //Lambda is a parameter defining the amount of regularization during filtering. Used by cv::ximgproc::DisparityWLSFilter(Default: 8000)

```

```c++

int  sigma_  //sigma determines the sensitivity of filtering. Used by cv::ximgproc::DisparityWLSFilter(Default: 5)

```

```c++

float  focus_  //focus value of the camera(Default: 1.3962)

```

```c++

float  baseline_  //Distance between centers of two stereo cameras(Default: 0.05)

```

```c++

bool  USE_FILTER  //Bool to determine if filter is used or not(Default: true)

```

```c++

cv::Mat  im_left_  //Stores left camera image

```

```c++

cv::Mat  im_right_  //Stores right camera image

```

```c++

ros::NodeHandle nh_private_  //ros::NodeHandle lets you specify a namespace for constructor

```

```c++

image_transport::ImageTransport it_  //wrapper for ros::NodeHandle that specialises for image data.

```

```c++

image_transport::ImageTransport it_private_  //wrapper for ros::NodeHandle that specialises for image data.

```
## Function references

#### depth_generator (1/2)
```c++
depth_generator(const ros::NodeHandle& nh,
			    const std::string& left_image_topic = "image1",
			    const std::string& right_image_topic = "image2")
```


### Parameters:

- **nh** : Global Node Handle object for node instantiation. No need to pass a private nodehandle, the class uses that by default.

- **left_image_topic** : image topic to subscribe to the left image frame. Default value is set to “image1”.

- **right_image_topic** : image topic to subscribe to the right image frame. Default value is set to “image2”.
------

#### depth_generator (2/2)
```c++
depth_generator(const ros::NodeHandle& nh,
				const std::map<std::string, float>& params,
				const std::string& image_left_topic = "image1",
				const std::string& image_right_topic = "image2")
```
  

### Parameters:

- **nh** : Global Node Handle object for node instantiation. No need to pass a private nodehandle, this class uses that by default.
    
- **map** : Stores key-value(string-float) pairs of parameters used for creating depth, disparity map and filtering. It should have the following entries.
    

    1. focus  (focal length)

    2. baseline (baseline distance)

    3. block_size  (kernel size for disparity matcher)

    4. num_disparities  (number of disparities output)

    5. lambda  (description)

    6. sigma  (description)

    7. USE_FILTER  (Flag: if set to 1, then it uses a filter to smoothen disparity maps, set it to 0 to turn that off.)

- **left_image_topic** : image topic to subscribe to the left image frame. Default value is set to “image1”.
    
- **right_image_topic** : image topic to subscribe to the right image frame. Default value is set to “image2”.
--------------------- 
    
### calc_depth()

```c++

void calc_depth(cv::Mat& depth = cv::Mat(),
				cv::Mat& disparity = cv::Mat())

```


### Parameters:

-   **depth** : (Optional) Output array to get a float matrix containing depth information.
    
-   **disparity** : (Optional) Output array to get a float matrix cotaining disparity map.

