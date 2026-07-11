import great_expectations as gx
import pandas as pd
import sys


def validate(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])

    context = gx.get_context(mode="file")  # contexte persistant -> génère gx/uncommitted/data_docs/

    # Datasource + asset pandas pointant sur notre DataFrame en mémoire
    data_source = context.data_sources.add_pandas("cacao_source")
    data_asset = data_source.add_dataframe_asset(name="cacao_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("cacao_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Suite d'attentes (on écrase si elle existe déjà)
    suite = gx.ExpectationSuite(name="prix_cacao_suite")
    suite = context.suites.add_or_update(suite)

    # --- 5 contraintes de qualité obligatoires ---
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="prix_xof"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="prix_xof", min_value=500, max_value=20000))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="humidite_pct", min_value=0, max_value=100))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="date"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="pays_origine", value_set=["Côte d'Ivoire", "Ghana", "Nigéria", "Cameroun"]))

    # Checkpoint = validation + génération des data docs
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name="cacao_checkpoint",
            validation_definitions=[
                gx.ValidationDefinition(
                    name="cacao_validation",
                    data=batch_definition,
                    suite=suite,
                )
            ],
            actions=[gx.checkpoint.actions.UpdateDataDocsAction(name="update_data_docs")],
        )
    )

    results = checkpoint.run(batch_parameters={"dataframe": df})

    n_ok = sum(r.success for r in results.run_results[list(results.run_results)[0]].results)
    print(f"{'Validation OK' if results.success else 'Validation ÉCHOUÉE'} "
          f"({n_ok}/5 expectations passées)")

    return results.success


if __name__ == '__main__':
    validate(sys.argv[1])
