// CSRT object tracking example (C) 2021 Stephane Charette <stephanecharette@gmail.com>
// MIT license applies.  See "license.txt" for details.


#include <opencv2/opencv.hpp>


/** These next few variables would be in a structure that gets passed around.
 * For ths simple example code I didn't make a structure and have them as globals.
 * @{
 */
std::chrono::high_resolution_clock::duration frame_duration;
cv::VideoCapture cap;
size_t fps_rounded = 0;
size_t total_frames = 0;
size_t total_wait_in_milliseconds = 0;
cv::Size desired_size(1024, 768);
std::string window_title = "CSRT Example";
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


/// Pause on the very first frame and reset the video to the start.
void pause_on_first_frame()
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

	std::cout << "Press any key to start.." << std::endl;
	cv::imshow(window_title, mat);
	cv::waitKey(-1);
}


/// Loop through the entire video, showing every frame.  Press @p ESC to exit, any other key to pause.
void show_video()
{
	auto time_to_show_next_frame = std::chrono::high_resolution_clock::now();

	// read the video and display each frame
	size_t frame_counter = 0;
	while (true)
	{
		cv::Mat mat;
		cap >> mat;
		if (mat.empty())
		{
			std::cout << "-> finished showing " << frame_counter << " frames" << std::endl;
			break;
		}

		frame_counter ++;
		if (frame_counter == total_frames or frame_counter % fps_rounded == 0)
		{
			const double average_wait_in_milliseconds = static_cast<double>(total_wait_in_milliseconds) / static_cast<double>(fps_rounded);

			std::cout
				<< "-> processing frame # " << frame_counter << "/" << total_frames
				<< " (" << (100.0 * frame_counter / total_frames) << "%), "
				<< "average pause is " << average_wait_in_milliseconds << " milliseconds"
				<< std::endl;
			total_wait_in_milliseconds = 0;
		}

		if (mat.size() != desired_size)
		{
			cv::Mat tmp;
			cv::resize(mat, tmp, desired_size);
			mat = tmp;
		}

		const auto now = std::chrono::high_resolution_clock::now();
		const auto milliseconds_to_pause = std::chrono::duration_cast<std::chrono::milliseconds>(time_to_show_next_frame - now).count();
		total_wait_in_milliseconds += milliseconds_to_pause;
		if (milliseconds_to_pause > 0)
		{
			// too early to show the next frame, we may need to briefly pause
			auto key = cv::waitKey(milliseconds_to_pause);
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
		pause_on_first_frame();
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
