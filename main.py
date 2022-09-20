import pandas as pd
from tabulate import tabulate


def get_cum_sum_pnl():
    pnl = df.groupby(['strategy', 'date'])['pos_pnl'].sum().to_frame().reset_index()
    pnl['cum_sum_pnl'] = pnl.groupby(['strategy'])['pos_pnl'].cumsum()
    return pnl


def get_gmv():
    return df.groupby(['strategy', 'date'])['pos_value'].apply(
        lambda c: c.abs().sum())


def get_daily_return():
    pnl = df.groupby(['strategy', 'date'])['pos_pnl'].sum()
    return pnl.div(gmv).reset_index().rename(columns={0: 'return'}).dropna(subset=['return'], how="all")


def get_cumulative_return():
    returns = get_daily_return()
    returns['cum_sum_return'] = returns.groupby(['strategy'])['return'].cumsum().mul(100).round(2)
    return returns.drop(columns=['return'])


def get_abs_drawdowns():
    cum_return = get_cumulative_return()
    cum_return['cum_max_return'] = cum_return.groupby(['strategy'])['cum_sum_return'].cummax()
    return cum_return.groupby(['strategy', 'date']).apply(
        lambda x: abs(x['cum_sum_return'] - x['cum_max_return'])).to_frame().reset_index().drop(
        columns=['level_2']).rename(columns={0: "drawdown"})


def get_drawdowns():
    cum_return = get_cumulative_return()
    cum_return['cum_max_return'] = cum_return.groupby(['strategy'])['cum_sum_return'].cummax()
    return cum_return.groupby(['strategy', 'date']).apply(
        lambda x: x['cum_sum_return'] - x['cum_max_return']).to_frame().reset_index().drop(
        columns=['level_2']).rename(columns={0: "drawdown"})


def add_dd_duration(dd):
    dd['is_dd'] = dd['drawdown'].apply(lambda c: False if c == 0.0 else True)
    dd['cumsum'] = dd.groupby('strategy')['is_dd'].cumsum()
    dd['duration'] = dd['cumsum'] - dd['cumsum'].where(~dd['is_dd']).ffill().fillna(0).astype(int)
    dd = dd.drop(columns=['is_dd', 'cumsum'])
    return dd


def get_prev_largest_dd(dd):
    ddowns = dd.groupby(['strategy']).tail(1)
    ddowns['date_overview'] = ddowns['date'] - pd.to_timedelta(ddowns['duration'], unit='d')
    ddowns = ddowns.drop(columns=['date', 'drawdown', 'duration'])
    largest_dd = dd.merge(ddowns, how='left', on='strategy')
    return largest_dd.loc[(largest_dd['date'] < largest_dd['date_overview']), ['strategy', 'drawdown']].groupby(
        ['strategy']).max().abs().reset_index().rename(columns={'drawdown': 'prev_drawdown'})


def detect_violated_largest_dd():
    tolerance = 1.2
    largest_dd = pd.merge(prev_largest_dd, current_dd, on=["strategy"])
    largest_dd['threshold'] = largest_dd['prev_drawdown'].apply(lambda row: row * tolerance)
    largest_dd['value'] = largest_dd['drawdown'].where(
        (largest_dd['drawdown'] >= largest_dd['threshold']) & largest_dd['threshold'] != 0)
    return merge_violated_data_with_gmv_pnl(largest_dd.dropna())


def get_prev_longest_dd(dd):
    durations = dd.groupby(['strategy']).tail(1)
    durations['date_overview'] = durations['date'] - pd.to_timedelta(durations['duration'], unit='d')
    durations = durations.drop(columns=['date', 'drawdown', 'duration'])
    longest_dd = dd.merge(durations, how='left', on='strategy')
    return longest_dd.loc[(longest_dd['date'] < longest_dd['date_overview']), ['strategy', 'duration']].groupby(
        ['strategy']).max().abs().reset_index().rename(columns={'drawdown': 'prev_drawdown'})


def detect_violated_absolute_current_dd():
    max_pnl = cum_sum_pnl
    max_pnl['cum_max_pnl'] = cum_sum_pnl.groupby(['strategy'])['cum_sum_pnl'].cummax().drop(columns=['cum_sum_pnl'])
    max_pnl = max_pnl.loc[(max_pnl['date'] == current_date), ['strategy', 'cum_max_pnl']]
    dd = pd.merge(current_pnl, max_pnl, on=["strategy"])
    dd = dd.groupby(['strategy']).apply(
        lambda x: x['cum_sum_pnl'] - x['cum_max_pnl']).to_frame().reset_index().drop(
        columns=['level_1']).rename(columns={0: 'drawdown'})
    absolute_current_drawdown_limit = 10000000
    dd['threshold'] = absolute_current_drawdown_limit
    violated = dd.loc[(abs(dd['drawdown']) > dd['threshold']), [
        'strategy', 'drawdown', 'threshold']]
    violated = violated.rename(
        columns={
            'drawdown': 'value',
        })
    return merge_violated_data_with_gmv_pnl(violated)


def detect_violated_eqy_dd():
    current_dd['threshold'] = 5
    violated = current_dd.loc[(current_dd['drawdown'] > current_dd['threshold']), ['strategy', 'drawdown', 'threshold']]
    violated = violated.rename(
        columns={
            'drawdown': 'value',
        })
    return merge_violated_data_with_gmv_pnl(violated)


def detect_longest_dd():
    tolerance = 1.5
    curr_dd = drawdowns.groupby(['strategy']).tail(1).rename(columns={'duration': 'curr_duration'})
    longest_dd = pd.merge(prev_longest_dd, curr_dd, on=["strategy"])
    longest_dd['threshold'] = longest_dd['duration'].apply(lambda row: row * tolerance)
    longest_dd['value'] = longest_dd['duration'].where(
        (longest_dd['curr_duration'] >= longest_dd['threshold']) & longest_dd['threshold'] != 0)
    return merge_violated_data_with_gmv_pnl(longest_dd.dropna())


def merge_violated_data_with_gmv_pnl(violated_data):
    violated_data = pd.merge(left=violated_data, right=current_gmv, left_on='strategy', right_on='strategy')
    violated_data = pd.merge(left=violated_data, right=current_pnl, left_on='strategy', right_on='strategy')
    return violated_data


def assemble_reports_row(violated_data, test_name):
    if violated_data.empty:
        return [], [], [], []

    violated_strategies = violated_data['strategy'].tolist()
    violated_values = violated_data['value'].tolist()
    test_names = [test_name] * len(violated_strategies)
    thresholds = violated_data['threshold'].tolist()
    gmvs = violated_data['pos_value'].tolist()
    cumu_pnls = violated_data['cum_sum_pnl'].tolist()
    return violated_strategies, test_names, violated_values, thresholds, gmvs, cumu_pnls


def print_report():
    strategies = []
    test_names = []
    current_values = []
    thresholds = []
    cur_gmv = []
    cur_cumu_pnl = []

    strategies_rows, test_names_rows, current_value_rows, threshold_rows, cur_gmv_rows, cur_cumu_pnl_rows \
        = assemble_reports_row(violated_absolute_current_dd, 'AbsDrawDown')
    strategies.extend(strategies_rows)
    test_names.extend(test_names_rows)
    current_values.extend(current_value_rows)
    thresholds.extend(threshold_rows)
    cur_gmv.extend(cur_gmv_rows)
    cur_cumu_pnl.extend(cur_cumu_pnl_rows)

    strategies_rows, test_names_rows, current_value_rows, threshold_rows, cur_gmv_rows, cur_cumu_pnl_rows \
        = assemble_reports_row(violated_eqy_dd, 'EQYDrawDown')
    strategies.extend(strategies_rows)
    test_names.extend(test_names_rows)
    current_values.extend(current_value_rows)
    thresholds.extend(threshold_rows)
    cur_gmv.extend(cur_gmv_rows)
    cur_cumu_pnl.extend(cur_cumu_pnl_rows)
    strategies_rows, test_names_rows, current_value_rows, threshold_rows, cur_gmv_rows, cur_cumu_pnl_rows \
        = assemble_reports_row(violated_largest_dd, 'LargestDrawDown')
    strategies.extend(strategies_rows)
    test_names.extend(test_names_rows)
    current_values.extend(current_value_rows)
    thresholds.extend(threshold_rows)
    cur_gmv.extend(cur_gmv_rows)
    cur_cumu_pnl.extend(cur_cumu_pnl_rows)
    strategies_rows, test_names_rows, current_value_rows, threshold_rows, cur_gmv_rows, cur_cumu_pnl_rows \
        = assemble_reports_row(violated_longest_dd, 'LongestDrawDown')
    strategies.extend(strategies_rows)
    test_names.extend(test_names_rows)
    current_values.extend(current_value_rows)
    thresholds.extend(threshold_rows)
    cur_gmv.extend(cur_gmv_rows)
    cur_cumu_pnl.extend(cur_cumu_pnl_rows)

    report = pd.DataFrame({
        'Strategy': strategies,
        'Test Violated': test_names,
        'Threshold Violated': thresholds,
        'Current Value that Violates': current_values,
        'Current GMV': cur_gmv,
        'Current CumuPNL': cur_cumu_pnl
    })
    print(tabulate(report, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('use_inf_as_na', True)
    pd.options.mode.chained_assignment = None
    df = pd.read_csv('sample_data.csv')
    current_date = 20151101
    cum_sum_pnl = get_cum_sum_pnl()
    current_pnl = cum_sum_pnl.loc[(cum_sum_pnl['date'] == current_date), ['strategy', 'cum_sum_pnl']]
    gmv = get_gmv()
    current_gmv = gmv.to_frame().reset_index().loc[
        (cum_sum_pnl['date'] == current_date), ['strategy', 'pos_value']]
    violated_absolute_current_dd = detect_violated_absolute_current_dd()
    drawdowns = get_abs_drawdowns()
    current_dd = drawdowns.loc[(drawdowns['date'] == current_date), ['strategy', 'drawdown']]
    current_dd['drawdown'] = abs(current_dd['drawdown'])
    violated_eqy_dd = detect_violated_eqy_dd()
    drawdowns = add_dd_duration(drawdowns)
    drawdowns['date'] = pd.to_datetime(drawdowns['date'].astype(str), format='%Y%m%d')
    prev_longest_dd = get_prev_longest_dd(drawdowns)
    violated_longest_dd = detect_longest_dd()
    prev_largest_dd = get_prev_largest_dd(drawdowns)
    violated_largest_dd = detect_violated_largest_dd()
    print_report()
