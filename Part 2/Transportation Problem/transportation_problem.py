import pulp


def data_midwest():
    S = ["Kansas City", "Omaha", "Davenport"]
    s = {"Kansas City": 150, "Omaha": 175, "Davenport": 275}
    D = ["Chicago", "St. Louis", "Cincinnati"]
    d = {"Chicago": 200, "St. Louis": 100, "Cincinnati": 300}
    c = {
            "Kansas City": {"Chicago": 6, "St. Louis": 8, "Cincinnati": 10},
            "Omaha": {"Chicago": 7, "St. Louis": 11, "Cincinnati": 11},
            "Davenport": {"Chicago": 4, "St. Louis": 5, "Cincinnati": 12},
        }

    return S, s, D, d, c


def data_airfreight():
    raw_data = [
        {
            "City": "Shanghai",
            "Country": "China",
            "Longitude": 121.5,
            "Latitude": 31.4,
            "Demand": 428,
            "Supply": None
        },
        {
            "City": "Singapore",
            "Country": "Singapore",
            "Longitude": 103.8,
            "Latitude": 1.3,
            "Demand": 122,
            "Supply": None
        },
        {
            "City": "Ningbo-Zhoushan",
            "Country": "China",
            "Longitude": 121.8,
            "Latitude": 29.9,
            "Demand": 335,
            "Supply": None
        },
        {
            "City": "Shenzhen (Yantian)",
            "Country": "China",
            "Longitude": 114.2,
            "Latitude": 22.6,
            "Demand": 183,
            "Supply": None
        },
        {
            "City": "Guangzhou (Nansha)",
            "Country": "China",
            "Longitude": 113.4,
            "Latitude": 22.7,
            "Demand": 378,
            "Supply": None
        },
        {
            "City": "Hong Kong",
            "Country": "Hong Kong SAR",
            "Longitude": 114.2,
            "Latitude": 22.3,
            "Demand": 189,
            "Supply": None
        },
        {
            "City": "Tianjin",
            "Country": "China",
            "Longitude": 117.8,
            "Latitude": 39.2,
            "Demand": 482,
            "Supply": None
        },
        {
            "City": "Busan",
            "Country": "South Korea",
            "Longitude": 129,
            "Latitude": 35.1,
            "Demand": 204,
            "Supply": None
        },
        {
            "City": "Qingdao",
            "Country": "China",
            "Longitude": 120.3,
            "Latitude": 36.1,
            "Demand": 218,
            "Supply": None
        },
        {
            "City": "Dubai (Jebel Ali)",
            "Country": "UAE",
            "Longitude": 55.2,
            "Latitude": 25,
            "Demand": 292,
            "Supply": None
        },
        {
            "City": "Rotterdam",
            "Country": "Netherlands",
            "Longitude": 4.5,
            "Latitude": 51.9,
            "Demand": 200,
            "Supply": None
        },
        {
            "City": "Antwerp",
            "Country": "Belgium",
            "Longitude": 4.4,
            "Latitude": 51.2,
            "Demand": 345,
            "Supply": None
        },
        {
            "City": "Laem Chabang",
            "Country": "Thailand",
            "Longitude": 100.4,
            "Latitude": 13.1,
            "Demand": 483,
            "Supply": None
        },
        {
            "City": "Los Angeles",
            "Country": "USA",
            "Longitude": -118.2,
            "Latitude": 33.7,
            "Demand": 95,
            "Supply": None
        },
        {
            "City": "Long Beach",
            "Country": "USA",
            "Longitude": -118.2,
            "Latitude": 33.8,
            "Demand": 152,
            "Supply": None
        },
        {
            "City": "Hamburg",
            "Country": "Germany",
            "Longitude": 9.9,
            "Latitude": 53.5,
            "Demand": 373,
            "Supply": None
        },
        {
            "City": "Kobe",
            "Country": "Japan",
            "Longitude": 135.2,
            "Latitude": 34.7,
            "Demand": 428,
            "Supply": None
        },
        {
            "City": "Barcelona",
            "Country": "Spain",
            "Longitude": 2.2,
            "Latitude": 41.3,
            "Demand": 455,
            "Supply": None
        },
        {
            "City": "Santos",
            "Country": "Brazil",
            "Longitude": -46.3,
            "Latitude": -23.9,
            "Demand": 321,
            "Supply": None
        },
        {
            "City": "Port Klang (Klang)",
            "Country": "Malaysia",
            "Longitude": 101.4,
            "Latitude": 3,
            "Demand": None,
            "Supply": 802
        },
        {
            "City": "Dalian",
            "Country": "China",
            "Longitude": 121.4,
            "Latitude": 38.9,
            "Demand": None,
            "Supply": 744
        },
        {
            "City": "Xiamen",
            "Country": "China",
            "Longitude": 118.1,
            "Latitude": 24.5,
            "Demand": None,
            "Supply": 420
        },
        {
            "City": "Kaohsiung",
            "Country": "Taiwan",
            "Longitude": 120.3,
            "Latitude": 22.6,
            "Demand": None,
            "Supply": 902
        },
        {
            "City": "Tanjung Priok",
            "Country": "Indonesia",
            "Longitude": 106.9,
            "Latitude": -6.2,
            "Demand": None,
            "Supply": 162
        },
        {
            "City": "Constanța",
            "Country": "Romania",
            "Longitude": 28.6,
            "Latitude": 44.2,
            "Demand": None,
            "Supply": 398
        },
        {
            "City": "Gdansk",
            "Country": "Poland",
            "Longitude": 18.7,
            "Latitude": 54.4,
            "Demand": None,
            "Supply": 272
        },
        {
            "City": "Vladivostok",
            "Country": "Russia",
            "Longitude": 131.9,
            "Latitude": 43.1,
            "Demand": None,
            "Supply": 612
        },
        {
            "City": "Dar es Salaam",
            "Country": "Tanzania",
            "Longitude": 39.2,
            "Latitude": -6.8,
            "Demand": None,
            "Supply": 918
        },
        {
            "City": "Colombo",
            "Country": "Sri Lanka",
            "Longitude": 79.9,
            "Latitude": 6.9,
            "Demand": None,
            "Supply": 282
        },
        {
            "City": "Gwangyang (South Korea)",
            "Country": "South Korea",
            "Longitude": 127.7,
            "Latitude": 34.9,
            "Demand": None,
            "Supply": 171
        }
    ]

    # Supply nodes
    S = []

    # Supply quantity
    s = {}

    # Demand nodes
    D = []

    # Demand quantity 
    d = {}

    # Lon/Lat data
    city_location = {}

    # Process raw data
    for entry in raw_data:
        if entry['Supply'] is not None:
            S.append(entry['City'])
            s[entry['City']] = entry['Supply']
        else:
            D.append(entry['City'])
            d[entry['City']] = entry['Demand']
        
        city_location[entry['City']] = {'lon': entry['Longitude'], 
                                        'lat': entry['Latitude']}

    # Shippping costs
    import haversine
    c = {
        i: {j: round(haversine.haversine(point1=(city_location[i]['lat'], city_location[i]['lon']),
                                         point2=(city_location[j]['lat'], city_location[j]['lon'])
                                         ), 2) 
                                         for j in D}
        for i in S
    } 
    
    return S, s, D, d, c


def transportation_problem_model(S, s, D, d, c):

    # Initialize our model
    myModel = pulp.LpProblem(name="Transportation Problem", sense=pulp.LpMinimize)

    # Define decision variables
    x = pulp.LpVariable.dicts(name="x", indices=(S, D), lowBound=0)

    # Objective
    myModel += pulp.lpSum(c[i][j] * x[i][j] for i in S for j in D)

    # Supply Constraint
    for i in S:
        myModel += s[i] == pulp.lpSum(x[i][j] for j in D), f"Supply_Capacity_{i}"

    # Demand Constraint
    for j in D:
        myModel += d[j] == pulp.lpSum(x[i][j] for i in S), f"Demand_Quantity_{j}"

    myModel.solve()


S, s, D, d, c = data_midwest()
S, s, D, d, c = data_airfreight()
transportation_problem_model(S, s, D, d, c)