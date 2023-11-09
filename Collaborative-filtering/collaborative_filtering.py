from person_scores import person_scores

def find_mean_score(person):
    """
    Takes in the key of a person in the person_scores dictionary 
    and returns the mean score of that person
    """

    scores : dict = person_scores[person]
    
    mean_score = sum( scores.values() ) / len(scores)

    return mean_score


def create_mean_scores_dictionary():
    """
    Returns a dictionary with the mean scores of each person
    """

    mean_scores = {}

    for person in person_scores:
        mean_scores[person] = find_mean_score(person)

    return mean_scores


def find_variation(person, show):
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
