from ticl.cli_parsing import make_model_level_argparser
from ticl.fit_model import main


def test_fit_model_help():
    try:
        main(['--help'])
    except SystemExit as e:
        assert e.code == 0
    else:
        assert False, "Expected SystemExit"


def test_mothernet_parser_exposes_grande_diversity_loss_weight():
    parser = make_model_level_argparser()
    namespace, _ = parser.parse_known_args(["mothernet"])

    assert hasattr(namespace.mothernet, "grande_diversity_loss_weight")
    assert namespace.mothernet.grande_diversity_loss_weight == 0.0
