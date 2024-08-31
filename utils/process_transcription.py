from utils.nlp_utils import match_segments, initialize_model_and_tokenizer
import utils.utils as utils
import re
import datetime


def process_transcription(
    transcription_response, file_name, translate_handler, target_language="en"
):
    """Process each transcription result for one alternative and translate."""
    translated_results = {}
    segment_times = []
    for i, result in enumerate(transcription_response.results):
        for index, alternative in enumerate(result.alternatives):
            original_segments = split_text(alternative.transcript)
            print(f"original {original_segments}")
            translated_text = translate_handler.translate_text(
                alternative.transcript, target_language
            )
            translated = split_text(translated_text)
            if len(original_segments) != len(translated):

                tokenizer, model = initialize_model_and_tokenizer()

                translated_segments = match_segments(
                    original_segments, translated, tokenizer, model
                )
                utils.save_object_to_pickle(
                    translated_segments,
                    f"/Users/ramiibrahimi/Documents/test/pkl/{file_name}_{i}_{index}_translated_segments.pkl",
                )
                # utils.load_object_from_pickle(
                #     f"/Users/ramiibrahimi/Documents/test/pkl/{file_name}_{i}_{index}_translated_segments.pkl"
                # )
            else:
                translated_segments = translated

            print(f"translated: {translated_segments}")
            if index == 0:
                segment_times = find_segment_starting_ending_times(
                    original_segments, alternative.words
                )
            timestamps = adjust_timestamps(translated_segments, segment_times)
            print(f"timestamps {timestamps}")
            alternative_key = f"alternative {index + 1}"

            if alternative_key not in translated_results:
                translated_results[alternative_key] = []

            translated_results[alternative_key].append(
                {
                    "transcript": alternative.transcript,
                    "translated_text": translated_segments,
                    "timestamps": timestamps,
                }
            )
    return translated_results


def find_segment_starting_ending_times(segments, words):
    """
    Find the starting and ending times for each segment based on the words' timestamps.

    :param segments: List of text segments split from the transcript.
    :param words: List of dictionaries with 'start_time' and 'end_time' for each word.
    :return: List of dictionaries with 'start_time' and 'end_time' for each segment.
    """
    segment_times = []
    current_word_index = 0

    # Iterate through each segment
    for segment in segments:
        # Find the number of words in the segment
        words_in_segment = segment.split()
        if current_word_index < len(words):
            # Fetch the start time of the first word in this segment
            segment_start_time = words[current_word_index].start_offset.total_seconds()
            if segment_start_time == 0:
                segment_start_time = words[
                    current_word_index
                ].end_offset.total_seconds()

            # Calculate the end word index for this segment
            end_word_index = current_word_index + len(words_in_segment) - 1

            # Make sure we do not go out of bounds
            if end_word_index < len(words):
                # Fetch the end time of the last word in this segment
                segment_end_time = words[end_word_index].end_offset.total_seconds()
            else:
                # If out of bounds, use the end time of the last available word
                segment_end_time = words[-1].end_offset.total_seconds()

            # Append the start and end times to the list
            segment_times.append(
                {"start_time": segment_start_time, "end_time": segment_end_time}
            )

            # Update the word index to the next word after the current segment
            current_word_index += len(words_in_segment)

    return segment_times


def split_text(translated_text):
    # Regular expression to split on '.', ',', '?', '!'
    # The pattern looks for these characters followed by one or more whitespace characters.
    # It keeps the punctuation as part of the previous segment.
    segments = re.split(r"(?<=[.,?!])\s+", translated_text)
    return segments


def adjust_timestamps(segments, segment_times, reading_speed=20):
    """
    Adjust the duration each segment should be displayed based on reading speed and actual word timings.

    :param segments: List of text segments.
    :param segment_times: List of dictionaries with 'start_time' and 'end_time' for each segment.
    :param reading_speed: Number of characters per second that can be comfortably read.
    :return: List of dictionaries with adjusted 'start_time' and 'end_time' for each segment.
    """
    adjusted_times = []
    prior_end_time = 0.0  # Initialize to 0 or the start of your video if known
    for segment, times in zip(segments, segment_times):
        segment_length = len(segment)
        estimated_duration = round(segment_length / reading_speed, 1)
        actual_start_time = times["start_time"]
        actual_end_time = times["end_time"]
        # actual_duration = actual_end_time - actual_start_time

        # # Ensure there is no overlap and no large gap
        # adjusted_start_time = max(prior_end_time, actual_start_time)

        # # Ensure the segment is displayed long enough
        # adjusted_end_time = adjusted_start_time + max(
        #     actual_duration, estimated_duration
        # )

        # # Update prior_end_time for the next segment
        # prior_end_time = adjusted_end_time

        adjusted_times.append(
            {
                "start_time": format_time(actual_start_time),
                "end_time": format_time(actual_end_time),
            }
        )

    return adjusted_times


def format_time(time_in_seconds):
    """Format time in seconds to H:M:S,MS."""
    ms = int((time_in_seconds - int(time_in_seconds)) * 1000)
    time = datetime.timedelta(seconds=int(time_in_seconds))
    return f"{str(time)},{ms:03d}"
