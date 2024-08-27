if __name__ == '__main__':
    # {'id': 2544, 'full_name': 'LeBron James', 'first_name': 'LeBron', 'last_name': 'James', 'is_active': True}
    all_teams = teams.get_teams()
    all_players = players.get_players()
    players_json = []
    for t in all_teams:
        name = t['full_name']
        print(name)
        if name == 'Golden State Warriors' or name == 'Boston Celtics' or name == 'Toronto Raptors':
            roster = commonteamroster.CommonTeamRoster(team_id=t['id']).get_dict()
            players_data = roster['resultSets'][0]
            headers = players_data['headers']
            row_set = players_data['rowSet']

            players_json = []
            for row in row_set:
                player_dict = dict(zip(headers, row))
                player_dict['team_name'] = name
                print(player_dict)
                meaningful_data = {
                    'team_name': name,
                    'player_id': player_dict['PLAYER_ID'],
                    'player_name': player_dict['PLAYER'],
                    'player_number': player_dict['NUM'],
                    'player_position': player_dict['POSITION'],
                    'player_school': player_dict['SCHOOL'],
                }
                players_json.append(meaningful_data)
            print(len(players_json))
            players_json.extend(players_json)
    script_dir = Path(__file__).parent
    filename = script_dir / 'current_nba_roster.json'
    print(players_json)
    with open(filename, 'w') as f:
        json.dump(players_json, f, indent=2)
