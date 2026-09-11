import unittest


class TestStructureFacades(unittest.TestCase):
    def test_service_and_contract_boundaries_import(self):
        from packages.contracts import ResponsesRequest
        from services.agent_hub import Serve
        from services.agent_hub.serve import Serve as ServeEntry
        from services.vault_gateway import VaultGateway

        self.assertIsNotNone(Serve)
        self.assertIs(Serve, ServeEntry)
        self.assertIsNotNone(VaultGateway)
        self.assertEqual(ResponsesRequest(input=[]).input, [])

    def test_pipeline_boundaries_import_without_running_external_work(self):
        from pipelines.content import extract_bvid
        from pipelines.content.bili import SOURCE as bili_source
        from pipelines.content.article import SOURCE as article_source
        from pipelines.intake import InboxQueueStore, classify
        from pipelines.intake.inbox import SOURCE as inbox_source

        self.assertEqual(
            extract_bvid("https://www.bilibili.com/video/BV1abcdefgh1"),
            "BV1abcdefgh1",
        )
        self.assertEqual(classify("https://www.bilibili.com/video/BV1abcdefgh1"),
                         ("bili", "https://www.bilibili.com/video/BV1abcdefgh1"))
        self.assertIsNotNone(InboxQueueStore)
        self.assertTrue(bili_source.is_file())
        self.assertTrue(article_source.is_file())
        self.assertTrue(inbox_source.is_file())


if __name__ == "__main__":
    unittest.main()
