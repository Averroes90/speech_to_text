from logging_config.custom_loggers import get_conditional_debug_logger,get_debug_logger
from typing import Optional
import spacy
from spacy.language import Language
import re
from itertools import zip_longest
import utils.utils as utils
import stanza

debug_mode_global= False
i=5
comp_word1= 'а'
comp_word2 = comp_word1
c_logger = get_conditional_debug_logger(f"{__name__}.conditional", index=i, word1=comp_word1, word2=comp_word2)
r_logger = get_debug_logger(f"{__name__}.debug")

# def load_nlp_model(language_code: str = "it") -> Language:
#     """
#     Loads a spaCy NLP model based on the specified language code.

#     Args:
#         language_code (str): A two-letter language code (e.g., 'it' for Italian, 'de' for German).

#     Returns:
#         Language: A spaCy Language object representing the loaded model.

#     Raises:
#         ValueError: If the specified language is unsupported.
#     """
#     model_names = {
#         "it": "it_core_news_lg",
#         "de": "de_core_news_lg",
#         "ru": "ru_core_news_lg",
#         "fr": "fr_core_news_sm",
#         "es": "es_core_news_sm",
#     }

#     if language_code in model_names:
#         model_name = model_names[language_code]
#     else:
#         raise ValueError("Unsupported language")

#     nlp = spacy.load(model_name)
#     return nlp


# def segment_text(text: str, nlp: Language) -> list[str]:
#     text = text.strip()
#     doc = nlp(text)
#     segments = [
#         sent.text.strip() for sent in doc.sents
#     ]  # Capture each sentence as a segment
#     return segments

def load_nlp_model(language_code: str) -> stanza.Pipeline:
    """
    Loads a Stanza NLP model with only the tokenizer for the specified language.

    Args:
        language_code (str): Language code (e.g., 'en' for English, 'sr' for Serbian).

    Returns:
        stanza.Pipeline: A Stanza Pipeline object for sentence segmentation.

    Raises:
        ValueError: If the specified language is unsupported.
    """
    # Check if the language is supported and download the model
    supported_languages = ["en", "it", "de", "ru", "fr", "es", "sr"]  # Extend this list based on your needs
    if language_code not in supported_languages:
        raise ValueError("Unsupported language")

    # Ensures the model is downloaded and loads only the tokenizer
    stanza.download(language_code, verbose=False)
    nlp = stanza.Pipeline(language_code, processors='tokenize')
    return nlp

def segment_text(text: str, nlp: stanza.Pipeline) -> list[str]:
    """
    Segments a text into sentences using a Stanza NLP model, suitable for subtitles.

    Args:
        text (str): The input text.
        nlp (stanza.Pipeline): A Stanza Pipeline object.

    Returns:
        list[str]: A list of sentence strings.
    """
    doc = nlp(text)
    return [sentence.text.strip() for sentence in doc.sentences]

def preprocess_and_tokenize(text: str) -> list[str]:
    """
    Remove all punctuation from the given text, convert to lowercase, and tokenize the text,
    removing any kind of whitespace characters.

    Args:
    - text (str): The text to process.

    Returns:
    - list[str]: A list of tokens with all punctuation removed, converted to lowercase, and all whitespace removed.
    """
    # Convert text to lowercase for uniformity
    text = text.lower()

    # Define a comprehensive list of punctuation characters
    punctuation = "!\"#$%&'()*+,./:;<=>?@[\\]`{|}~"

    # Remove all punctuation
    cleaned_text = "".join(char for char in text if char not in punctuation)

    # Use regex to split on any whitespace and remove empty strings from the result
    tokens = [
        token.strip() for token in re.split(r"\s+", cleaned_text) if token.strip()
    ]

    return tokens


def find_last_index(lst: list[any], value: any, start: int, end: int) -> int:
    """
    Finds the last index of 'value' in 'lst' between 'start' and 'end'.

    Args:
    lst (list): The list to search.
    value (str): The value to search for.
    start (int): The starting index for the search.
    end (int): The ending index for the search.

    Returns:
    int: The last index of the found value, or -1 if not found.
    """
    # print(lst)
    # print(end)
    # print(start)
    for i in range(end, start - 1, -1):  # Search from end to start
        if lst[i] == value:
            return i
    return -1


def fill_missing_pairs(
    input_dict: dict[int, int], len_words1: int, len_words2: int
) -> dict[int, int]:
    result_dict = {}
    prev_key, prev_value = -1, -1

    for key, value in input_dict.items():
        if (
            prev_key != -1
            and key - prev_key > 1
            and value - prev_value == key - prev_key
        ) or (prev_key == -1 and key == value and key > 0):
            # Fill in missing key-value pairs
            for i in range(prev_key + 1, key):
                result_dict[i] = prev_value + (i - prev_key)
        result_dict[key] = value
        prev_key, prev_value = key, value
    # Fill missing pairs at the end
    if (
        prev_key != -1
        and prev_key < len_words2 - 1
        and len_words2 - prev_key == len_words1 - prev_value
    ):
        for i in range(prev_key + 1, len_words2):
            result_dict[i] = prev_value + (i - prev_key)

    return result_dict


def match_transcripts(words1: list[str], words2: list[str]) -> dict[int, int]:
    index_mapping = {}
    margin = 5  # Margin for approximate matching position
    # print(f"words1 {words1}")
    # print(f"words2 {words2}")
    # Forward matching: from the start to the middle of words1
    for i in range(len(words1)):
        word_first = words1[i]
        word_last = words1[-(i + 1)]

        # Search range for the first word
        start_range = max(i - margin, 0)
        end_range = min(
            i + margin + 1, len(words2)
        )  # end range should be inclusive, hence +1

        # Matching the first word in the approx range in words2
        if word_first in words2[start_range:end_range]:
            first_match_index = words2.index(word_first, start_range, end_range)
            index_mapping[first_match_index] = i

        # Reverse matching: from the end towards the middle of words1
        # Calculate reverse index for words2 that mirrors the position in words1
        reverse_index = len(words1) - 1 - i
        start_range_last = max(reverse_index - margin, 0)
        end_range_last = min(
            reverse_index + margin + 1, len(words2)
        )  # end range should be inclusive, hence +1

        if word_last in words2[start_range_last:end_range_last]:
            # print(start_range_last)
            # print(end_range_last)
            last_match_index = find_last_index(
                words2, word_last, end_range_last, start_range_last
            )
            if last_match_index != -1:
                index_mapping[last_match_index] = reverse_index

    # Sorting index_mapping by the keys (indices from words2)
    sorted_index_mapping = dict(sorted(index_mapping.items()))

    # Assuming fill_missing_pairs is a function you've defined elsewhere to fill gaps
    enhanced_index_mapping = fill_missing_pairs(
        sorted_index_mapping, len(words1), len(words2)
    )

    return enhanced_index_mapping


def clone_timestamps(
    transcript1: str,
    transcript2: str,
    timestamp_mapping: dict[int, dict[str, Optional[float]]],
) -> dict[str, Optional[str], Optional[float], Optional[float]]:
    #######################
    ##for dubugging
    comp_word_1_index = 0
    comp_word_2_index = 0
    #############################
    words1 = preprocess_and_tokenize(transcript1)
    words2 = preprocess_and_tokenize(transcript2)
    #############
    extras = {'word1':words1[comp_word_1_index],'word2':words2[comp_word_2_index]}
    c_logger.debug(f"words1: {words1}",extra=extras)
    c_logger.debug(f"words2: {words2}",extra=extras)
    c_logger.debug(f"len w1: {len(words1)}",extra=extras)
    c_logger.debug(f"len w2: {len(words2)}",extra=extras)
    c_logger.debug(f"ts map: {timestamp_mapping}",extra=extras)
    c_logger.debug(f"ts map len : {len(timestamp_mapping)}",extra=extras)
    c_logger.debug(f"last word words1 {words1[-1]}",extra=extras)
    c_logger.debug(f"last word words2 {words2[-1]}",extra=extras)
    #######################

    adjusted_timestamp_mapping = fill_missing_words(
        words=words1, timestamps=timestamp_mapping
    )

    # #############
    # if debug_mode and (
    #     words1[comp_word_1_index] == comp_word_1
    #     or words2[comp_word_2_index] == comp_word_2
    # ):
    #     print(f"adj ts map: {adjusted_timestamp_mapping}")
    #     print(f"adj ts map len: {len(adjusted_timestamp_mapping)}")
    c_logger.debug(f"adj ts map: {adjusted_timestamp_mapping}",extra=extras)
    c_logger.debug(f"adj ts map len: {len(adjusted_timestamp_mapping)}",extra=extras)
    # #######################
    index_mapping = match_transcripts(words1, words2)
    # #############
    # if debug_mode and (
    #     words1[comp_word_1_index] == comp_word_1
    #     or words2[comp_word_2_index] == comp_word_2
    # ):
    #     print(f"index map: {index_mapping}")
    #     print(f"index map len: {len(index_mapping)}")
    c_logger.debug(f"index map: {index_mapping}",extra=extras)
    c_logger.debug(f"index map len: {len(index_mapping)}",extra=extras)
    #######################

    # Initialize the timestamp list for words2 with default values
    words2_w_timestamps = []

    # Ensure there is at least one word in both lists to avoid index errors
    if words1 and words2:
        # Assign timestamps to the first word in words2 from the first word in words1
        words2_w_timestamps.append(
            {
                "word2": words2[0],
                'word1': words1[0] if words1[0] == words2[0] else None,
                "start_time": adjusted_timestamp_mapping[0]["start_time"],
                "end_time": adjusted_timestamp_mapping[0]["end_time"],
            }
        )

        # Map timestamps based on index_mapping for the middle words
        for i, word in enumerate(words2):
            if (i == 0) or (i == len(words2) - 1):  # Skip the first and last words
                continue
            # print(f"word_in_ words2: {word}")
            if i in index_mapping:
                word1_index = index_mapping[i]
                words2_w_timestamps.append(
                    {
                        "word2": word,
                        "word1": words1[word1_index],
                        "start_time": adjusted_timestamp_mapping[word1_index][
                            "start_time"
                        ],
                        "end_time": adjusted_timestamp_mapping[word1_index]["end_time"],
                    }
                )
            else:
                # If no mapping found, you might want to handle it, e.g., no timestamp or approximate
                words2_w_timestamps.append(
                    {"word2": word, "word1": None, "start_time": None, "end_time": None}
                )

        # Assign timestamps to the last word in words2 from the last word in words1
        last_index1 = len(adjusted_timestamp_mapping) - 1
        last_index2 = len(words2) - 1
        ###################################################################
        # if debug_mode and (
        #     words1[comp_word_1_index] == comp_word_1
        #     or words2[comp_word_2_index] == comp_word_2
        # ):
        #     last_index1
        #     print(f"adjusted time stampping {adjusted_timestamp_mapping}")
        #     print(f"adjusted time stampping  last {adjusted_timestamp_mapping[last_index1]["end_time"]}")
        #     print(f"last index1 {last_index1}")
        #     print(f"last index2 {last_index2}")
        #     print(f"words 2 in mapping {words2}")
        #     print(f"words 1 in mapping {words1}")
        #     print(f"len of words 2 in mapping {len(words2)}")
        #     print(f"len of adjusted mapping {len(adjusted_timestamp_mapping)}")
        c_logger.debug(f"adjusted time stampping {adjusted_timestamp_mapping}",extra=extras)
        c_logger.debug(f"adjusted time stampping  last {adjusted_timestamp_mapping[last_index1]["end_time"]}",extra=extras)
        c_logger.debug(f"last index1 {last_index1}",extra=extras)
        c_logger.debug(f"last index2 {last_index2}",extra=extras)
        c_logger.debug(f"words 2 in mapping {words2}",extra=extras)
        c_logger.debug(f"words 1 in mapping {words1}",extra=extras)
        c_logger.debug(f"len of words 2 in mapping {len(words2)}",extra=extras)
        c_logger.debug(f"len of adjusted mapping {len(adjusted_timestamp_mapping)}",extra=extras)
         ###################################################################
        if last_index2 in index_mapping:
            start_time = adjusted_timestamp_mapping[last_index1]["start_time"]
        else:
            start_time = None
        words2_w_timestamps.append(
            {
                "word2": words2[last_index2],
                "word1": words1[last_index1],
                "start_time": start_time,
                "end_time": adjusted_timestamp_mapping[last_index1]["end_time"],
            }
        )
        ##################################
        # if debug_mode and (
        #     words1[comp_word_1_index] == comp_word_1
        #     or words2[comp_word_2_index] == comp_word_2
        # ):
        #     print(words2_w_timestamps)
        c_logger.debug(f"words2_w_timestamps: {words2_w_timestamps}",extra=extras)
        ##################################################

    return words2_w_timestamps


def find_segment_starting_ending_times(
    segments: list[str], stamped_transcript: dict[str, str, float, float]
) -> list[dict[str, float, str, float]]:
    segment_times = []
    current_word_index = 0

    for segment in segments:
        # print(f"segment {segment}")
        words_in_segment = preprocess_and_tokenize(segment)
        if current_word_index >= len(stamped_transcript):
            # print(f"Ran out of timestamps, cannot process segment: {segment}")
            break  # Stop processing if there are no more timestamps
        segment_start_time = stamped_transcript[current_word_index]["start_time"]

        end_word_index = current_word_index + len(words_in_segment) - 1
        if end_word_index >= len(stamped_transcript):
            # print(f"Missing timestamp data for segment: {segment}")
            segment_end_time = stamped_transcript[len(stamped_transcript) - 1][
                "end_time"
            ]  # Last known good timestamp
        else:
            segment_end_time = stamped_transcript[end_word_index]["end_time"]

        segment_times.append(
            {"start_time": segment_start_time, "end_time": segment_end_time}
        )
        # print(f"start time {segment_start_time}")
        # print(f"end time {segment_end_time}")
        current_word_index += len(words_in_segment)

    return segment_times


def combine_short_segments(
    segments: list[str],
    segment_times: list[dict[str, float]],
    min_length: int = 10,
) -> list[tuple[str, float, float]]:
    combined_segments = []
    current_combination = ""
    current_combination_start_time = None
    segment_end_time = None
    next_start_time = None  # Define outside to ensure scope availability

    for i, (segment, segment_time) in enumerate(zip(segments, segment_times)):
        if current_combination == "":
            current_combination = segment
            current_combination_start_time = segment_time["start_time"]
        else:
            current_combination += " " + segment
        segment_end_time = segment_time["end_time"]
        if i < min(len(segments) - 1, len(segment_times) - 1):  # not final loop
            next_start_time = segment_times[i + 1]["start_time"]

            if (
                len(current_combination) > min_length
                and next_start_time
                and segment_end_time
            ):
                combined_segments.append(
                    (
                        current_combination,
                        current_combination_start_time,
                        segment_end_time,
                    )
                )
                current_combination = ""
                next_start_time = None
        else:  # final loop
            combined_segments.append(
                (
                    current_combination,
                    current_combination_start_time,
                    segment_end_time,
                )
            )
    # Handle remaining segments if segments are longer than times
    if segments and segment_times and len(segments) > len(segment_times):
        if combined_segments and current_combination:
            # Combine remaining segments into the last entry
            remaining_segments = " ".join(segments[len(segment_times) :])
            current_combination += " " + remaining_segments
            combined_segments[-1] = (
                current_combination,
                current_combination_start_time,
                segment_end_time,
            )
        elif not combined_segments:
            # If no segments were combined yet, handle all remaining as a new entry
            remaining_segments = " ".join(segments[len(segment_times) :])
            combined_segments.append(
                (remaining_segments, current_combination_start_time, segment_end_time)
            )

    return combined_segments


def combine_on_none(
    segments: list[str], segment_times: list[dict[str, float]]
) -> list[tuple[str, float, float]]:
    combined_segments = []
    current_combination = ""
    combined_seg_start_time = None
    next_segment_start_time = None
    # if segments[0] == "Угу.":

    #     print(f"inside combine on none pre segments {segments}")
    #     print(f"inside combine on none pre segment times {segment_times}")
    for index, (segment, segment_time) in enumerate(
        zip_longest(
            segments, segment_times, fillvalue={"start_time": None, "end_time": None}
        )
    ):
        # ##################################################################
        # #for debugging
        # if segments[0] == "Угу.":
        #     print(f"index: {index}")
        #     print(f'segments: {segments}')
        #     print(f"segment_times {segment_times}")
        #     print(f'sssssssegments: {segment}')
        #     print(f"sssssssssegment_time {segment_time}")
        # ###################################################################
        if not current_combination:  # new combo
            combined_seg_start_time = segment_time["start_time"]
            current_combination = segment
        else:
            current_combination += " " + segment

        if index < len(segment_times) - 1:
            next_segment_start_time = segment_times[index + 1]["start_time"]

        if (
            segment_time is not None
            and segment_time["end_time"] != None
            and current_combination
            and next_segment_start_time != None
        ):
            combined_segments.append(
                (current_combination, combined_seg_start_time, segment_time["end_time"])
            )
            current_combination = ""
            next_segment_start_time = None
            continue

    # Handle remaining segments if segments are longer than times
    if current_combination:
        combined_segments.append(
            (
                current_combination,
                combined_seg_start_time,
                segment_times[-1]["end_time"],
            )
        )
    # if segments[0] == "Молодец.":
    # print(f"inside combine on none combined segments {combined_segments}")
    return combined_segments


def process_chirp_responses(
    chirp_response: any,
    chirp_2_response: any,
    source_language: str,
    audio_duration: float = 10000,
) -> tuple[str, str]:
    ################
    ##for debugging
    debug_mode = debug_mode_global
    debug_index = 36
    #############################
    nlp = load_nlp_model(source_language)
    chirp_2_key = next(iter(chirp_2_response.results))
    chirp_key = next(iter(chirp_response.results))
    segments_w_stamps1 = []
    segments_w_stamps2 = []
    for i, result2 in enumerate(
        chirp_2_response.results[chirp_2_key].inline_result.transcript.results
    ):
        # print(f"loop 1 i: {i}")
        result1 = chirp_response.results[chirp_key].inline_result.transcript.results[i]
        if not (result2.alternatives) or (not result1.alternatives):
            # print(f"loop 1 skipped loop {i}")
            continue

        transcript1 = result1.alternatives[0].transcript
        # print(f"transcript 1 loop 1 {transcript1}")
        transcript2 = result2.alternatives[0].transcript
        # print(f"transcript 2 loop 1 {transcript2}")
        transcript2 = pick_better_transcript(
            transcript1=transcript1, transcript2=transcript2
        )
        word_timestamp_mapping = extract_words_timings(result1)
        ################
        # for debugging
        # if debug_mode and i == debug_index:
        #     print(
        #         f"result {i} word time stamp map pre clone len{len(word_timestamp_mapping)} word_timestamp_map {word_timestamp_mapping}"
        #     )
        #     print(f"result {i} trans1 pre clone {transcript1}")
        #     print(f"result {i} trans2 pre clone {transcript2}")
        extras = {'index':debug_index}
        c_logger.debug(f"result {i} word time stamp map pre clone len{len(word_timestamp_mapping)} word_timestamp_map {word_timestamp_mapping}",extra=extras)
        c_logger.debug(f"result {i} trans1 pre clone {transcript1}",extra=extras)
        c_logger.debug(f"result {i} trans2 pre clone {transcript2}",extra=extras)
        # ################
        stamped_transcript2 = clone_timestamps(
            transcript1, transcript2, word_timestamp_mapping
        )
        # for debugging
        ###############
        # if debug_mode and i == debug_index:
        #     print(
        #         f"result {i} stamped transcript pre segmentation post clone len{len(stamped_transcript2)} segment_transcript2 {stamped_transcript2}"
        #     )
        #     print(f"result {i} transcript2 pre segmentation post clone {transcript2}")
        c_logger.debug(f"result {i} stamped transcript pre segmentation post clone len{len(stamped_transcript2)} segment_transcript2 {stamped_transcript2}",extra=extras)
        c_logger.debug(f"result {i} transcript2 pre segmentation post clone {transcript2}",extra=extras)
        ################
        segments2 = segment_text(transcript2, nlp)
        # for debugging
        ###################
        # if debug_mode and i == debug_index:
        #     print(f"result {i} segments2 post segmentation {segments2}")
        c_logger.debug(f"result {i} segments2 post segmentation {segments2}",extra=extras)
        ################

        segment_stamps2 = find_segment_starting_ending_times(
            segments2, stamped_transcript2
        )
        # ###########################
        # if debug_mode and i == debug_index:
        #     print(
        #         f"post find starting and ending times loop1 segments2 len{len(segments2)} segments2 {segments2}"
        #     )
        #     print(
        #         f"post find starting and ending times loop1 segment stamps 2 len{len(segment_stamps2)} segment_stamps2 {segment_stamps2}"
        #     )
        c_logger.debug(f"post find starting and ending times loop1 segments2 len{len(segments2)} segments2 {segments2}",extra=extras)
        c_logger.debug(f"post find starting and ending times loop1 segment stamps 2 len{len(segment_stamps2)} segment_stamps2 {segment_stamps2}",extra=extras)
        #############################
        combined_segments2 = combine_on_none(
            segments=segments2, segment_times=segment_stamps2
        )
        combined_segments1 = segment_chirp_1_transcript(
            combined_segments2, transcript1, transcript2
        )
        segments_w_stamps2.extend(combined_segments2)
        segments_w_stamps1.extend(combined_segments1)
    # print(segments_w_stamps1)
    # print(segments_w_stamps2)
    r_logger.info(f'segments_w_stamps1: {segments_w_stamps1}')
    r_logger.info(f'segments_w_stamps2: {segments_w_stamps2}')
    srt_subtitles1 = utils.create_srt(segments_w_stamps1, audio_duration)
    srt_subtitles2 = utils.create_srt(segments_w_stamps2, audio_duration)
    return srt_subtitles1, srt_subtitles2


def segment_chirp_1_transcript(
    chirp_2_segments: list[str, float, float],
    chirp_1_transcript: str,
    chirp_2_transcript: str,
) -> list[str, float, float]:
    words1 = preprocess_and_tokenize(chirp_1_transcript)
    words2 = preprocess_and_tokenize(chirp_2_transcript)
    index_mapping = match_transcripts(words1, words2)

    word1_start_index = 0
    word1_end_index = 0
    word2__prior_end_index = 0
    chirp_1_segments = []

    for segment in chirp_2_segments:
        words2 = preprocess_and_tokenize(segment[0])
        word2_end_index = len(words2) - 1 + word2__prior_end_index
        for j in range(0, word2_end_index + 1):
            if (word2_end_index - j) in index_mapping:
                word1_end_index = index_mapping[word2_end_index - j]
                break

        chirp_1_segments.append(
            (
                " ".join(words1[word1_start_index : word1_end_index + 1]),
                segment[1],
                segment[2],
            )
        )
        word1_start_index = word1_end_index + 1
        word2__prior_end_index += len(words2)
    return chirp_1_segments


def extract_words_timings(result: any) -> dict[int, dict[str, Optional[float]]]:

    alternative = result.alternatives[0]
    words = alternative.words
    # print(f"result {result}")
    # print(f"words1 in extract {words}")
    # print(f"alternative {alternative}")

    word_info = {
        i: {
            "word": preprocess_and_tokenize(word.word)[0] if preprocess_and_tokenize(word.word) else word.word,
            "start_time": word.start_offset.total_seconds(),
            "end_time": word.end_offset.total_seconds(),
        }
        for i, word in enumerate(words)
    }
    if (
        result.result_end_offset.total_seconds()
        != word_info[len(word_info) - 1]["end_time"]
    ):
        print(f"missmatched endtime between last word and result for{result} ")
        word_info[len(word_info) - 1][
            "end_time"
        ] = result.result_end_offset.total_seconds()
    return word_info


# def fill_missing_words(
#     words: list[str], timestamps: dict[int, dict[str, Optional[float]]]
# ) -> dict[int, dict[str, Optional[float]]]:
#     """
#     Align timestamps with a list of words. If a word does not have a timestamp, it will be assigned None for start and end times.

#     Args:
#     - words (List[str]): List of words.
#     - timestamps (Dict[int, Dict[str, Optional[float]]]): Dictionary of timestamps indexed by integer with word, start_time, and end_time.

#     Returns:
#     - Dict[int, Dict[str, Optional[float]]]: Dictionary where each integer index corresponds to a dictionary containing the word, start time, and end time.
#     """
#     aligned_timestamps = {}
#     adjusted_index = 0

#     for index, word in enumerate(words):
#         word3 = preprocess_and_tokenize(timestamps[index - adjusted_index]["word"])
#         if word3[0] == word:
#             aligned_timestamps[index] = {
#                 "word": word,
#                 "start_time": timestamps[index - adjusted_index]["start_time"],
#                 "end_time": timestamps[index - adjusted_index]["end_time"],
#             }
#         else:
#             adjusted_index += 1
#             aligned_timestamps[index] = {
#                 "word": word,
#                 "start_time": None,
#                 "end_time": None,
#             }


#     return aligned_timestamps


def fill_missing_words(
    words: list[str], timestamps: dict[int, dict[str, Optional[float]]]
) -> dict[int, dict[str, Optional[float]]]:
    """
    Align timestamps with a list of words. If a word does not have a timestamp, it will be assigned None for start and end times.

    Args:
    - words (List[str]): List of words.
    - timestamps (Dict[int, Dict[str, Optional[float]]]): Dictionary of timestamps indexed by integer with word, start_time, and end_time.

    Returns:
    - Dict[int, Dict[str, Optional[float]]]: Dictionary where each integer index corresponds to a dictionary containing the word, start time, and end time.
    """
    aligned_timestamps = {}
    # print(f"inside fill missing word time stamps{timestamps}")
    # print(f"inside fill missing words {words}")
    # print(f"len timestamps {len(timestamps)}")
    # print(f"len wordss {len(words)}")
    n_missing = 0

    for index, word in enumerate(words):
        if (index - n_missing) in timestamps:
            word_from_timestamps = timestamps[index - n_missing]["word"]
            word2 = preprocess_and_tokenize(word_from_timestamps)[0]
            # #################################
            # if timestamps[0]['word'] == "супер":
            #     print(
            #         f"Index: {index} Word: {word} Processed TS Word: {word2} TS Word: {word_from_timestamps}"
            #     )
            #     print(f"wwwwwwwooooorrrrrd {word}")
            #     print(f"ts index minus n missing {timestamps[index - n_missing]["word"]}")
            # ####################################

            if word2 == word:
                aligned_timestamps[index] = timestamps[index - n_missing]
            else:
                n_missing += 1
                if index > 0 and index < len(words)-1:
                    aligned_timestamps[index] = {
                        "word": word,
                        "start_time": None,
                        "end_time": None,
                    }
                elif index == 0: #first word is missing
                    aligned_timestamps[index] = {
                        "word": word,
                        "start_time": timestamps[0]['start_time'],
                        "end_time": None,
                    }
                elif index == len(words) -1: #last word is missing
                    aligned_timestamps[index] = {
                        "word": word,
                        "start_time": None,
                        "end_time": timestamps[len(timestamps)-1]['end_time'],
                    }
        else:
            # print(f"No timestamp entry for index {index} word: {word}")
            if index > 0 and index < len(words)-1:
                aligned_timestamps[index] = {
                    "word": word,
                    "start_time": None,
                    "end_time": None,
                }
            elif index == 0: #first word is missing
                aligned_timestamps[index] = {
                    "word": word,
                    "start_time": timestamps[0]['start_time'],
                    "end_time": None,
                }
            elif index == len(words) -1: #last word is missing
                aligned_timestamps[index] = {
                    "word": word,
                    "start_time": None,
                    "end_time": timestamps[len(timestamps)-1]['end_time'],
                }


    # if timestamps[0]['word'] == "супер":
    #     print(f"number of missing {n_missing}")
    #     print(f"len timestamps {len(timestamps)}")
    #     print(f"len adj timestamps {len(aligned_timestamps)}")
    return aligned_timestamps


# the logic here can be expanded in the future
def pick_better_transcript(transcript1: str, transcript2: str) -> str:
    better_transcript = ""
    words1 = preprocess_and_tokenize(transcript1)
    words2 = preprocess_and_tokenize(transcript2)

    # if chirp has a transcript and chirp 2 doesnt, take chirp1
    if words1 and not words2:
        better_transcript = transcript1
    else:
        better_transcript = transcript2

    return better_transcript


# def fix_none_stamps(
#     index: int, subtitles: list[tuple[str, float, float]]
# ) -> tuple[str, float, float]:
#     new_text, new_start, new_end = subtitles[index]
#     text, start, end = subtitles[index]
#     print(f"attempting to fix a none start or none end at index {index}")

#     if index > 0:
#         prior_text, prior_start, prior_end = subtitles[index - 1]
#         if start is None:
#             new_start = prior_end
#     else:
#         prior_text, prior_start, prior_end = None, None, None
#         if start is None:
#             new_start = max(end - 4, 0) if end is not None else 0
#     if index < len(subtitles) - 1:
#         next_text, next_start, next_end = subtitles[index + 1]
#         if end is None:
#             end = next_start
#     else:
#         next_text, next_start, next_end = None, None, None
#         if end is None:
#             new_end = start + 4
#     return new_text, new_start, new_end


def fix_none_stamps(
    index: int, subtitles: list[tuple[str, float, float]], audio_duration: float = 10000
) -> tuple[str, float, float]:
    current_text, current_start, current_end = subtitles[index]
    print(f"attempting to fix a none start or none end at index {index}")
    # Attempt to fix a None start time
    if current_start is None:
        # Use the end time of the previous subtitle if possible
        if index > 0:
            _, _, prior_end = subtitles[index - 1]
            current_start = prior_end
        else:
            # If it's the first subtitle and start is None, assume a start time of 0
            current_start = max(current_end - 4, 0) if current_end is not None else 0

    # Attempt to fix a None end time
    if current_end is None:
        # Use the start time of the next subtitle if possible
        if index < len(subtitles) - 1:
            _, next_start, _ = subtitles[index + 1]
            current_end = next_start
        else:
            # If it's the last subtitle and end is None, add a default duration to start
            current_end = min(
                current_start + 4, audio_duration
            )  # Assume a default duration of 4 seconds

    return (current_text, current_start, current_end)
