"""
Mapping of Project Gutenberg IDs to book titles for all authors.
All titles verified directly from Project Gutenberg (2025-10-26).
"""

BOOK_TITLES = {
    # Jane Austen (7 books) - All verified ✓
    '105': 'Persuasion',
    '121': 'Northanger Abbey',
    '141': 'Mansfield Park',
    '158': 'Emma',
    '161': 'Sense and Sensibility',
    '1342': 'Pride and Prejudice',
    '946': 'Lady Susan',

    # L. Frank Baum - Oz series (14 books) - All verified ✓
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

    # Charles Dickens (14 books) - All verified ✓
    '98': 'A Tale of Two Cities',
    '580': 'The Pickwick Papers',
    '675': 'American Notes',
    '700': 'The Old Curiosity Shop',
    '730': 'Oliver Twist',
    '766': 'David Copperfield',
    '786': 'Hard Times',
    '821': 'Dombey and Son',
    '963': 'Little Dorrit',
    '967': 'Nicholas Nickleby',
    '968': 'Martin Chuzzlewit',
    '1023': 'Bleak House',
    '1400': 'Great Expectations',
    '24022': 'A Christmas Carol',

    # F. Scott Fitzgerald (8 books) - All verified ✓
    '4368': 'Flappers and Philosophers',
    '6695': 'Tales of the Jazz Age',
    '805': 'This Side of Paradise',
    '9830': 'The Beautiful and Damned',
    '64317': 'The Great Gatsby',
    '68229': 'All the Sad Young Men',
    'gutenberg_net_au_ebooks03_0301261': 'Tender Is the Night',
    'gutenberg_net_au_fsf_PAT-HOBBY': 'The Pat Hobby Stories',

    # Herman Melville (10 books) - All verified ✓
    '15': 'Moby-Dick; or, The Whale',
    '2694': 'I and My Chimney',
    '4045': 'Omoo: Adventures in the South Seas',
    '10712': 'White Jacket; Or, The World on a Man-of-War',
    '11231': 'Bartleby, the Scrivener: A Story of Wall-Street',
    '13720': 'Mardi, and a voyage thither, Vol. 1 (of 2)',
    '13721': 'Mardi, and a voyage thither, Vol. 2 (of 2)',
    '15422': 'Israel Potter: His Fifty Years of Exile',
    '21816': 'The Confidence-Man: His Masquerade',
    '28656': 'Typee',

    # Ruth Plumly Thompson - Oz series (13 books) - All verified ✓
    '53765': 'Kabumpo in Oz',
    '55806': 'Ozoplaning with the Wizard of Oz',
    '55851': 'The Wishing Horse of Oz',
    '56073': 'Captain Salt in Oz',
    '56079': 'Handy Mandy in Oz',
    '56085': 'The Silver Princess in Oz',
    '58765': 'The Cowardly Lion of Oz',
    '61681': 'Grampa in Oz',
    '65849': 'The Lost King of Oz',
    '70152': 'The Hungry Tiger of Oz',
    '71273': 'The Gnome King of Oz',
    '73170': 'The giant horse of Oz',
    '75720': 'Jack Pumpkinhead of Oz',

    # Mark Twain (6 books) - All verified ✓
    '74': 'The Adventures of Tom Sawyer, Complete',
    '76': 'Adventures of Huckleberry Finn',
    '86': 'A Connecticut Yankee in King Arthur\'s Court',
    '1837': 'The Prince and the Pauper',
    '3176': 'The Innocents Abroad',
    '3177': 'Roughing It',

    # H.G. Wells (12 books) - All verified ✓
    '35': 'The Time Machine',
    '36': 'The War of the Worlds',
    '159': 'The island of Doctor Moreau',
    '1047': 'The New Machiavelli',
    '1059': 'The World Set Free',
    '5230': 'The Invisible Man: A Grotesque Romance',
    '6424': 'A Modern Utopia',
    '12163': 'The Sleeper Awakes',
    '23218': 'The Red Room',
    '27365': 'Tales of Space and Time',
    '52501': 'The First Men in the Moon',
    '75786': 'The open conspiracy : Blue prints for a world revolution',
}


def get_book_title(filename):
    """Get book title from Gutenberg ID filename."""
    # Extract ID from filename (e.g., "54.txt" -> "54")
    gutenberg_id = filename.replace('.txt', '')
    return BOOK_TITLES.get(gutenberg_id, f'Project Gutenberg #{gutenberg_id}')
