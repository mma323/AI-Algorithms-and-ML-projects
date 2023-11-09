from person_scores import person_scores

def find_mean_score(person : str) -> float:
    """
    Takes in the key of a person in the person_scores dictionary 
    and returns the mean score of that person
    """

    scores : dict = person_scores[person]
    
    mean_score : float = sum( scores.values() ) / len(scores)

    return mean_score


def mean_scores() -> dict:
    """
    Returns a dictionary with the mean scores of each person
    by calling the find_mean_score function for each person
    in the person_scores dictionary and storing the results
    in a new dictionary
    """

    mean_scores = {}

    for person in person_scores:
        mean_scores[person] = find_mean_score(person)

    return mean_scores


def find_variation(person : str, show  : str) -> float:
    """
    Takes in the key of a person in the person_scores dictionary 
    and the name of a show the person has watched and returns the
    difference between the score the person gave to the show and
    the mean score of the person
    """

    scores : dict = person_scores[person]
    mean_score = find_mean_score(person)

    variation = scores[show] - mean_score

    return variation


def variations() -> dict:
    """
    Returns a dictionary of dictionaries with the variation of each show for each person
    by calling the find_variation function for each person and show
    in the person_scores dictionary and storing the results
    in a new dictionary of dictionaries
    """

    variations = {}

    for person in person_scores:
        variations[person] = {}
        for show in person_scores[person]:
            variations[person][show] = find_variation(person, show)

    return variations
