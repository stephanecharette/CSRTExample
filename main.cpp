// CSRT object tracking example (C) 2021 Stephane Charette <stephanecharette@gmail.com>
// MIT license applies.  See "license.txt" for details.


#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>


typedef cv::Ptr<cv::Tracker> Tracker;	///< single object tracker (could be any OpenCV tracker, not just CSRT)

struct ObjectTracker
{
	bool		is_valid;	///< used to detremine if this tracker should be used or skipped
	std::string	name;		///< name we give to the tracker for debug purposes
	cv::Scalar	colour;		///< colour we'll use to draw the output onto the mat
	cv::Rect2d	rect;		///< last reported rectangle for this tracker
	size_t		last_valid;	///< last frame index where this tracker reported positive results
	Tracker		tracker;	///< CSRT tracker

	/// Create an object Tracker from a rectangle and an image.
	ObjectTracker(const std::string n, const cv::Scalar c, const cv::Rect2d r, cv::Mat & mat) :
		is_valid(true),
		name(n),
		colour(c),
		rect(r),
		last_valid(0)
	{
		tracker = cv::TrackerCSRT::create();
		tracker->init(mat, rect);
		return;
	}

	/// Create an object Tracker from 4 normalized X,Y,W,H values instead of a cv::Rect2d.
	ObjectTracker(const std::string n, const cv::Scalar c, const double x, const double y, const double w, const double h, cv::Mat & mat) :
		ObjectTracker(n, c, cv::Rect2d(x * mat.cols, y * mat.rows, w * mat.cols, h * mat.rows), mat)
	{
		return;
	}
};


typedef std::vector<ObjectTracker> VObjectTrackers;


/** These next few variables would be in a structure or class that gets passed around.
 * For this example code I kept it simple and left them as globals.
 * @{
 */
std::chrono::high_resolution_clock::duration frame_duration;
cv::VideoCapture cap;
cv::Size desired_size(1024, 768);
bool enable_object_tracking				= true;
std::string window_title				= "CSRT Example";
size_t fps_rounded						= 0;
size_t total_frames						= 0;
/// @}

/// All trackers used while the video is being processed (people, ball, etc).
VObjectTrackers all_trackers;


/// Remember that OpenCV uses BGR, not RGB. @{
const cv::Scalar red	(0.0	, 0.0	, 255.0	);
const cv::Scalar blue	(255.0	, 0.0	, 0.0	);
const cv::Scalar green	(0.0	, 255.0	, 0.0	);
const cv::Scalar purple	(128.0	, 0.0	, 128.0	);
const cv::Scalar black	(0.0	, 0.0	, 0.0	);
const cv::Scalar white	(255.0	, 255.0	, 255.0	);
/// @}


/// Open the video, get the timing information we need, and display a few statistics.
void initialize_video(const std::string & filename)
{
	cap.open(filename);
	if (cap.isOpened() == false)
	{
		throw std::invalid_argument("failed to open " + filename);
	}

	const int width					= cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH	);
	const int height				= cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT	);
	const size_t number_of_frames	= cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT	);
	const double fps				= cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS			);
	const double fpm				= fps * 60.0;
	const double minutes			= std::floor(number_of_frames / fpm);
	const double seconds			= (number_of_frames - (minutes * fpm)) / fps;

	fps_rounded = std::round(fps);
	total_frames = number_of_frames;

	/* 1 second = 1000 milliseconds
	 * 1 second = 1000000 microseconds
	 * 1 second = 1000000000 nanoseconds
	 */

	const size_t length_of_each_frame_in_nanoseconds = std::round(1000000000.0 / fps);
	frame_duration = std::chrono::nanoseconds(length_of_each_frame_in_nanoseconds);

	std::cout	<< filename << ":" << std::endl
				<< "-> " << width << " x " << height << " @ " << fps << " FPS for "
				<< minutes << "m" << std::fixed << std::setprecision(1) << seconds << "s"
				<< " (" << number_of_frames << " total frames)" << std::endl
				<< "-> each frame is " << length_of_each_frame_in_nanoseconds << " nanoseconds"
				<< " (" << (length_of_each_frame_in_nanoseconds / 1000000.0) << " milliseconds)" << std::endl;

	// figure out how much we need to zoom each frame (if they're too big to display on my screen)
	double factor = 1.0;
	if (width > desired_size.width or height > desired_size.height)
	{
		// we're going to have to resize each frame since they're larger than what we want to see
		const double horizontal_factor	= static_cast<double>(desired_size.width)	/ static_cast<double>(width);
		const double vertical_factor	= static_cast<double>(desired_size.height)	/ static_cast<double>(height);
		factor							= std::max(horizontal_factor, vertical_factor);
		desired_size = cv::Size(std::round(factor * width), std::round(factor * height));

		std::cout
			<< "-> each frame will be resized to " << desired_size.width << " x " << desired_size.height
			<< " (zoom factor of " << factor << ")"
			<< std::endl;

	}
	else
	{
		// make the desired size match the frame dimensions so we don't resize anything
		desired_size = cv::Size(width, height);
	}
	window_title = window_title + " (" + std::to_string(width) + " x " + std::to_string(height) + " @ " + std::to_string(static_cast<int>(std::round(100.0 * factor))) + "%)";

	return;
}


cv::Mat get_first_frame()
{
	cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0.0);
	cv::Mat mat;
	cap >> mat;
	cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0.0);

	if (mat.size() != desired_size)
	{
		cv::Mat tmp;
		cv::resize(mat, tmp, desired_size);
		mat = tmp;
	}

	return mat;
}


/** Initialize the trackers with the coordinates of the objects we need to track.  Normally, the coordinates would need
 * to come from something else, like the output of a neural network.  But this example code doesn't have a neural network
 * or any other place where we get the coordinates.  Instead, this function has some hard-coded coordinates which I've
 * manually calculated beforehand as objects of interest so we can demo CSRT object tracking.
 */
void initialize_trackers(cv::Mat & mat, const std::string & filename)
{
	/* All coordinates in this function are normalized.  This allows the code to work regardless of the
	 * "desired size" set at the top of this file.  Once we multiply the desired by the normalized values
	 * we'll get the coordinates which OpenCV expects us to be using.
	 */

	if (filename.find("input_3733.mp4") != std::string::npos)	// 3 kids passing the ball on soccer field.  Tracker quickly loses track of the ball but maintains track on the kids.
	{
		all_trackers.emplace_back("ball", green	, 0.697435897, 0.539062500, 0.029304029, 0.052083333, mat);
		all_trackers.emplace_back("p1"	, red	, 0.704029304, 0.207031250, 0.083516484, 0.359375000, mat);
		all_trackers.emplace_back("p2"	, blue	, 0.032967033, 0.276041667, 0.122344322, 0.458333333, mat);
		all_trackers.emplace_back("p3"	, purple, 0.083516484, 0.087239583, 0.069597070, 0.272135417, mat);
	}
	else if (filename.find("input_3750.mp4") != std::string::npos)	// 2 kids on basekeball court.  Tracker loses the one in the background.
	{
		all_trackers.emplace_back("p1"	, red	, 0.565567766, 0.471354167, 0.099633700, 0.528645833, mat);
		all_trackers.emplace_back("p2"	, blue	, 0.441758242, 0.533854167, 0.070329670, 0.330729167, mat);
	}

	// go through the trackers again, this time to draw all the original rectangles onto the image
	for (auto & ot : all_trackers)
	{
		cv::rectangle(mat, ot.rect, ot.colour);
	}

	return;
}


/// Pause on the very first frame and reset the video to the start.
void pause_on_first_frame(cv::Mat & mat)
{
	std::cout << "Press any key to start.." << std::endl;
	cv::imshow(window_title, mat);
	cv::waitKey(-1);
}


/// Loop through the entire video, showing every frame.  Press @p ESC to exit, any other key to pause.
void show_video()
{
	size_t frame_counter = 0;
	auto time_to_show_next_frame = std::chrono::high_resolution_clock::now();
	auto previous_timestamp = time_to_show_next_frame;
	size_t previous_frame_counter = 0;

	// read the video and display each frame
	while (true)
	{
		cv::Mat mat;
		cap >> mat;
		if (mat.empty())
		{
			std::cout << "-> finished showing " << frame_counter << " frames" << std::endl;
			break;
		}

		// once per second we want to display some information on where we are and the FPS
		if (frame_counter + 1 >= total_frames or frame_counter % fps_rounded == 0)
		{
			const auto now				= std::chrono::high_resolution_clock::now();
			const auto duration			= now - previous_timestamp;
			const auto recent_frames	= frame_counter - previous_frame_counter;
			const auto fps				= 1000.0 * recent_frames / std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
			previous_frame_counter		= frame_counter;
			previous_timestamp			= now;

			std::cout
				<< "-> processing frame # " << frame_counter << "/" << total_frames
				<< " (" << (100.0 * (frame_counter + 1) / total_frames) << "%), "
				<< fps << " FPS"
				<< std::endl;
		}

		if (mat.size() != desired_size)
		{
			cv::Mat tmp;
			cv::resize(mat, tmp, desired_size);
			mat = tmp;
		}

		// now we update all the CSRT trackers
		for (auto & ot : all_trackers)
		{
			if (ot.is_valid)
			{
				// this next call takes a *LONG* time to run!
				const bool ok = ot.tracker->update(mat, ot.rect);
				if (ok)
				{
					ot.last_valid = frame_counter;
				}
				else
				{
					// we've lost the object...is it temporary?
					ot.rect = cv::Rect2d(-1.0, -1.0, -1.0, -1.0);

					if (frame_counter > ot.last_valid + fps_rounded * 3)
					{
						std::cout << "-> removing tracker for \"" << ot.name << "\" since object not seen since frame #" << ot.last_valid << "" << std::endl;
						ot.is_valid = false;
					}
				}
			}
		}

		// and finally we draw all the recent tracker rectangles onto the image
		for (auto & ot : all_trackers)
		{
			if (ot.last_valid == frame_counter)
			{
				cv::rectangle(mat, ot.rect, ot.colour);
			}
		}

		const auto now = std::chrono::high_resolution_clock::now();
		auto number_of_milliseconds_to_pause = std::chrono::duration_cast<std::chrono::milliseconds>(time_to_show_next_frame - now).count();
		if (number_of_milliseconds_to_pause <= 0)
		{
#if 1
			// For this example code only, we're going to wait a minimum of 1 millisecond.  This will ensure
			// that OpenCV gets time to redraw the window.  Otherwise we might not see anything since tracking
			// is so slow that we'll always be falling behind.
			number_of_milliseconds_to_pause = 1;
#endif
		}

		if (number_of_milliseconds_to_pause > 0)
		{
			// too early to show the next frame, we may need to briefly pause
			auto key = cv::waitKey(number_of_milliseconds_to_pause);
			if (key != -1 and key != 27)
			{
				// user has pressed a key -- assume they're asking to pause the video
				std::cout << "-> paused on frame #" << frame_counter << std::endl;
				key = cv::waitKey(-1);

				// we have no idea how long we paused, so reset the time point we used to control the frame rate
				time_to_show_next_frame = std::chrono::high_resolution_clock::now();
			}
			if (key == 27) // ESC
			{
				throw std::runtime_error("user requested to quit");
			}
		}
		cv::imshow(window_title, mat);
		time_to_show_next_frame += frame_duration;
		frame_counter ++;
	}

	return;
}


int main(int argc, char *argv[])
{
	try
	{
		std::string filename;
		if (argc > 1)
		{
			filename = argv[1];
		}
		else
		{
		}

		initialize_video(filename);
		cv::Mat mat = get_first_frame();
		if (enable_object_tracking)
		{
			initialize_trackers(mat, filename);
		}
		pause_on_first_frame(mat);
		show_video();

		// and pause again on the last frame which was shown
		std::cout << "Done! Press any key to exit." << std::endl;
		cv::waitKey(-1);
	}
	catch (const std::exception & e)
	{
		std::cout << "ERROR: " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cout << "ERROR: unknown exception caught" << std::endl;
		return 2;
	}

	return 0;
}
