"""
Mapping of Project Gutenberg IDs to book titles for all authors.
"""

BOOK_TITLES = {
    # L. Frank Baum - Oz series (14 books)
    '54': 'The Wonderful Wizard of Oz',
    '955': 'The Marvelous Land of Oz',
    '957': 'Ozma of Oz',
    '958': 'Dorothy and the Wizard in Oz',
    '959': 'The Road to Oz',
    '22566': 'The Emerald City of Oz',
    '26624': 'The Patchwork Girl of Oz',
    '30852': 'Tik-Tok of Oz',
    '33361': 'The Scarecrow of Oz',
    '39868': 'Rinkitink in Oz',
    '41667': 'The Lost Princess of Oz',
    '43936': 'The Tin Woodman of Oz',
    '50194': 'The Magic of Oz',
    '52176': 'Glinda of Oz',

    # Jane Austen (7 books)
    '105': 'Persuasion',
    '121': 'Northanger Abbey',
    '141': 'Mansfield Park',
    '158': 'Emma',
    '161': 'Sense and Sensibility',
    '1342': 'Pride and Prejudice',
    '946': 'Lady Susan',

    # Charles Dickens (14 books)
    '46': 'A Christmas Carol',
    '98': 'A Tale of Two Cities',
    '730': 'Oliver Twist',
    '766': 'David Copperfield',
    '786': 'The Pickwick Papers',
    '821': 'Dombey and Son',
    '883': 'Our Mutual Friend',
    '917': 'The Mystery of Edwin Drood',
    '963': 'Bleak House',
    '967': 'Little Dorrit',
    '1023': 'Hard Times',
    '1400': 'Great Expectations',
    '30127': 'Martin Chuzzlewit',
    '42232': 'Nicholas Nickleby',

    # F. Scott Fitzgerald (8 books)
    '64317': 'The Great Gatsby',
    '9830': 'This Side of Paradise',
    '805': 'The Beautiful and Damned',
    '4368': 'Tales of the Jazz Age',
    '2052': 'Tender Is the Night',
    '243': 'Flappers and Philosophers',
    '6695': 'All the Sad Young Men',
    '23032': 'The Curious Case of Benjamin Button',

    # Herman Melville (10 books)
    '2489': 'Moby-Dick',
    '10712': 'Bartleby, the Scrivener',
    '11231': 'Typee',
    '13720': 'Omoo',
    '15859': 'Mardi',
    '21816': 'Redburn',
    '10009': 'White-Jacket',
    '9147': 'Pierre',
    '8118': 'Israel Potter',
    '10641': 'Billy Budd, Sailor',

    # Ruth Plumly Thompson - Oz series (13 books)
    '48778': 'Speedy in Oz',
    '40455': 'The Yellow Knight of Oz',
    '40726': 'Pirates in Oz',
    '42054': 'The Purple Prince of Oz',
    '43710': 'Ojo in Oz',
    '45332': 'The Wishing Horse of Oz',
    '46450': 'Captain Salt in Oz',
    '47298': 'Handy Mandy in Oz',
    '49072': 'The Silver Princess in Oz',
    '50318': 'Ozoplaning with the Wizard of Oz',
    '52091': 'The Wonder City of Oz',
    '54261': 'The Scalawagons of Oz',
    '56851': 'Lucky Bucky in Oz',

    # Mark Twain (6 books)
    '76': 'Adventures of Huckleberry Finn',
    '74': 'The Adventures of Tom Sawyer',
    '86': 'The Prince and the Pauper',
    '119': 'A Connecticut Yankee in King Arthur\'s Court',
    '3176': 'The Gilded Age',
    '245': 'Life on the Mississippi',

    # H.G. Wells (12 books)
    '35': 'The Time Machine',
    '36': 'The War of the Worlds',
    '5230': 'The Invisible Man',
    '159': 'The Island of Doctor Moreau',
    '1743': 'The First Men in the Moon',
    '4032': 'In the Days of the Comet',
    '17': 'The Door in the Wall',
    '524': 'The Food of the Gods',
    '775': 'When the Sleeper Wakes',
    '718': 'The Wheels of Chance',
    '27': 'The World Set Free',
    '1743': 'A Modern Utopia',
}


def get_book_title(filename):
    """Get book title from Gutenberg ID filename."""
    # Extract ID from filename (e.g., "54.txt" -> "54")
    gutenberg_id = filename.replace('.txt', '')
    return BOOK_TITLES.get(gutenberg_id, f'Project Gutenberg #{gutenberg_id}')
