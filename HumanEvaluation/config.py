RecommendationTasks = [
    'ContributionRepo',
    'PRReviewer',
    'RepoMaintainer',
    'SimDeveloper'
]


PREFIX = {
    'ContributionRepo': './ContributionRepo/',
    'PRReviewer':'./PRReviewer/',
    'RepoMaintainer':'./RepoMaintainer/',
    'SimDeveloper':'./SimDeveloper/'
}

SUFFIX = {
    'ContributionRepo' : [
        'CF',
        'DTEM',
        'GAT',
        'Topic',
    ],
    'PRReviewer' : [
        'CF',
        'DTEM',
        'GAT'
    ],
    'RepoMaintainer' : [
        'CF',
        'DTEM',
        'GAT',
        'Topic',
    ],
    'SimDeveloper' : [
        'CF',
        'Dev2Vec',
        'DTEM',
        'GAT',
        'Topic',
    ]
}

SELECTED_DATA = {
    'ContributionRepo': './selected/ContributionRepo.json',
    'PRReviewer':'./selected/PRReviewer.json',
    'RepoMaintainer':'./selected/RepoMaintainer.json',
    'SimDeveloper':'./selected/SimDeveloper.json'
}

CONTRIBUTOR =   '../GNN/DataPreprocess/full_graph/content/contributors.json'
REPOSITORY =    '../GNN/DataPreprocess/full_graph/content/repositories.json'
PRS =           '../GNN/DataPreprocess/full_graph/content/prs.json'

RESULT_PREFIX = './result/result'
