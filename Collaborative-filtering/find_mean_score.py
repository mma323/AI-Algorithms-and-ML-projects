from person_scores import person_scores

def find_mean_score(person : str):
    """
    Find the mean score of a person by getting all of their scores from the
    person_scores dictionary and then dividing the sum of their scores by the
    number of scores they have.
    """
    
    scores : dict = person_scores[person]
    mean_score : int = sum( scores.values() ) / len(scores)

    return mean_score

