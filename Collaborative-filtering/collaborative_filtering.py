import numpy as np
from person_scores import person_scores, SHOWS


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
    Returns a dictionary of dictionaries with the variation 
    of each show for each person by calling the find_variation function 
    for each person and show in the person_scores dictionary 
    and storing the results in a new dictionary of dictionaries
    """

    variations = {}

    for person in person_scores:
        variations[person] = {}
        for show in person_scores[person]:
            variations[person][show] = find_variation(person, show)

    return variations


def pad_variation_vectors(variations : dict) -> None:
    """
    Returns a dictionary of dictionaries with the variation 
    of each show for each person by calling the variations function
    and adding "x" as variation for shows that the person has not watched
    """

    for person in variations:
        for show in SHOWS:
            if show not in variations[person]:
                variations[person][show] = "x"


def compute_correlation(
    active_person : str, other_person : str, variations : dict
) -> float:
    """
    Takes in the keys of two persons in the person_scores dictionary 
    and returns the correlation between the two persons
    """

    active_person_variations = []
    other_person_variations  = []

    for show in SHOWS:
        if (
            variations[active_person][show] != "x" and 
            variations[other_person][show] != "x"
        ):
            active_person_variations.append(variations[active_person][show])
            other_person_variations.append(variations[other_person][show])
    
    correlation = np.corrcoef(
        active_person_variations, other_person_variations
    )[0, 1]

    return correlation


def compute_correlations(active_person : str, variations : dict) -> dict:
    """
    Takes in the key of a person in the person_scores dictionary 
    and returns a dictionary with the correlation between the person
    and all other persons in the person_scores dictionary
    """

    correlations = {}

    for other_person in variations:
        if other_person != active_person:
            correlations[other_person] = (
                compute_correlation(active_person, other_person, variations)
            )

    return correlations


def predict_score(
        variations : dict , active_user : str, show : str, kappa : float = 1
) -> float:
    """
    Takes in the key of a person in the person_scores dictionary
    and the name of a show the person has not watched and returns
    the predicted score the person would give to the show
    """
    
    average_score_active_user            : float = find_mean_score(active_user)
    sum_of_correlation_variance_products : float = 0

    for other_user in variations:
        if other_user != active_user and variations[other_user][show] != "x":
            correlation = compute_correlation(
                active_user, other_user, variations
            )
            variation   = find_variation(other_user, show)
            sum_of_correlation_variance_products += correlation * variation

    predicted_score = (
        average_score_active_user +
        kappa * sum_of_correlation_variance_products
    )

    return predicted_score


def main():
    variation_vectors : dict = variations()

    pad_variation_vectors(variation_vectors)

    print(
        "Predicted score Westworld: ", 
        round( predict_score(variation_vectors, "new person", "westworld") , 2)
    )
    print(
        "Predicted score Skam: ", 
        round( predict_score(variation_vectors, "new person", "skam") , 2 )
    )


if __name__ == "__main__":
    main()