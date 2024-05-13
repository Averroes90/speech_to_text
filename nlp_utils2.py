from typing import Optional
import spacy
from spacy.language import Language
import re
from process_transcription import adjust_timestamps
import utils


def load_nlp_model(language_code: str = "it") -> Language:
    """
    Loads a spaCy NLP model based on the specified language code.

    Args:
        language_code (str): A two-letter language code (e.g., 'it' for Italian, 'de' for German).

    Returns:
        Language: A spaCy Language object representing the loaded model.

    Raises:
        ValueError: If the specified language is unsupported.
    """
    model_names = {
        "it": "it_core_news_lg",
        "de": "de_core_news_lg",
        "ru": "ru_core_news_lg",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
    }

    if language_code in model_names:
        model_name = model_names[language_code]
    else:
        raise ValueError("Unsupported language")

    nlp = spacy.load(model_name)
    return nlp


def segment_text(text: str, nlp: Language) -> list[str]:
    text = text.strip()
    doc = nlp(text)
    segments = [
        sent.text.strip() for sent in doc.sents
    ]  # Capture each sentence as a segment
    return segments


# def combine_short_sentences(sentences: list[str], min_length: int = 40) -> list[str]:
#     combined_sentences = []
#     temp_sentence = ""

#     for sentence in sentences:
#         if len(sentence) + len(temp_sentence) < min_length:
#             temp_sentence += " " + sentence
#         else:
#             if temp_sentence:
#                 combined_sentences.append(temp_sentence.strip())
#                 temp_sentence = sentence
#             else:
#                 temp_sentence = sentence

#     if temp_sentence:  # Add the last sentence if it's left out
#         combined_sentences.append(temp_sentence.strip())

#     return combined_sentences


# def segment_transcript(text: str, nlp: Language) -> list[str]:
#     segments = segment_text(text, nlp)
#     combined_segments = combine_short_sentences(segments)
#     return combined_segments


def preprocess_and_tokenize(transcript: str) -> list[str]:
    """
    Preprocesses the transcript by converting to lowercase and removing punctuation, then tokenizes it.

    Args:
    transcript (str): The original transcript text.

    Returns:
    list: A list of cleaned, tokenized words from the transcript.
    """
    # Convert to lowercase and remove punctuation
    cleaned_transcript = re.sub(r"[^\w\s]", "", transcript.lower())
    # Tokenize the transcript
    words = cleaned_transcript.split()
    return words


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
    # words1 = preprocess_and_tokenize(words1)
    # words2 = preprocess_and_tokenize(words2)
    # print(words1)
    # print(words2)
    # print(len(words1))
    # print(len(words2))
    index_mapping = {}
    start_index = 0
    end_index = len(words2) - 1

    for i in range(min(len(words1), len(words2)) // 2):
        word_first = words1[i]
        word_last = words1[-(i + 1)]

        if word_first in words2[start_index : end_index + 1]:
            first_match_index = words2.index(word_first, start_index, end_index + 1)
            index_mapping[first_match_index] = i
            start_index = first_match_index + 1

        if word_last in words2[start_index : end_index + 1]:
            last_match_index = find_last_index(
                words2, word_last, start_index, end_index
            )
            if last_match_index != -1:
                index_mapping[last_match_index] = len(words1) - 1 - i
                end_index = last_match_index - 1
    sorted_index_mapping = dict(sorted(index_mapping.items()))
    enhanced_index_mapping = fill_missing_pairs(
        sorted_index_mapping, len(words1), len(words2)
    )
    return enhanced_index_mapping


def clone_timestamps(
    transcript1: str,
    transcript2: str,
    timestamp_mapping: dict[int, dict[str, Optional[float]]],
) -> dict[str, Optional[str], Optional[float], Optional[float]]:
    words1 = preprocess_and_tokenize(transcript1)
    words2 = preprocess_and_tokenize(transcript2)
    # print(f"words1: {words1}")
    # print(f"words2: {words2}")
    # print(f"len w1: {len(words1)}")
    # print(f"len w2: {len(words2)}")
    adjusted_timestamp_mapping = fill_missing_wordss(
        words=words1, timestamps=timestamp_mapping
    )
    index_mapping = match_transcripts(words1, words2)
    # print(f"index_mapping: {index_mapping}")
    # Initialize the timestamp list for words2 with default values
    words2_w_timestamps = []

    # Ensure there is at least one word in both lists to avoid index errors
    if words1 and words2:
        # Assign timestamps to the first word in words2 from the first word in words1
        words2_w_timestamps.append(
            {
                "word": words2[0],
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
        # print(words2_w_timestamps)

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
            print(f"Missing timestamp data for segment: {segment}")
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
    segment_times: list[dict[str, float, str, float]],
    min_length: int = 10,
) -> list[str, float, float]:
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


def combine_on_none(segments, segment_times):
    combined_segments = []
    current_combination = ""
    for i, segment, segment_time in enumerate(zip(segments, segment_times)):

        if segment_time["end_time"] != None:
            combined_segments.append(segment)
            current_combination = ""
            continue

        if i < min(len(segments), len(segment_times)):
            current_combination += segments[i + 1]
            if segment_times[i + 1]["end_time"] != None:
                combined_segments.append(current_combination)
                current_combination = ""
                continue

    # Handle remaining segments if segments are longer than times
    if segments and segment_times and len(segments) > len(segment_times):
        combined_segments.append(
            current_combination + " " + " ".join(segments[len(segments) :])
        )

    return combined_segments


def process_chirp_responses(
    chirp_response: any, chirp_2_response: any, source_language: str
) -> tuple[str, str]:

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
        word_timestamp_mapping = extract_words_timings(result1)
        stamped_transcript2 = clone_timestamps(
            transcript1, transcript2, word_timestamp_mapping
        )
        segments2 = segment_text(transcript2, nlp)

        segment_stamps2 = find_segment_starting_ending_times(
            segments2, stamped_transcript2
        )
        # print(f"loop1 segments2 {segments2} segment stamps 2 {segment_stamps2}")
        combined_segments2 = combine_short_segments(
            segments=segments2, segment_times=segment_stamps2
        )
        combined_segments1 = segment_chirp_1_transcript(
            combined_segments2, transcript1, transcript2
        )
        segments_w_stamps2.extend(combined_segments2)
        segments_w_stamps1.extend(combined_segments1)
    # print(segments_w_stamps1)
    # print(segments_w_stamps2)
    srt_subtitles1 = utils.create_srt(segments_w_stamps1)
    srt_subtitles2 = utils.create_srt(segments_w_stamps2)
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
    # print(f"words1 in extract {words}")
    # print(f"alternative {alternative}")
    word_info = {
        i: {
            "word": word.word,
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


# def fill_missing_wordss(
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


def fill_missing_wordss(
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

    for index, word in enumerate(words):
        word2 = preprocess_and_tokenize(timestamps[index]["word"])
        if index in timestamps and word2 == word:
            aligned_timestamps[index] = timestamps[index]
        else:
            aligned_timestamps[index] = {
                "word": word,
                "start_time": None,
                "end_time": None,
            }

    return aligned_timestamps
