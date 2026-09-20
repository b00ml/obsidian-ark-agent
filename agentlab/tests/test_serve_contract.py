"""契约层单元测试（F5-007）：验证错误码、request_id、响应格式一致性"""

import unittest
from agentlab.runtime.serve_contract import (
    ErrorCode,
    StandardResponse,
    map_exception_to_error,
    _generate_request_id,
)


class TestErrorCode(unittest.TestCase):
    """测试错误码枚举"""
    
    def test_error_code_values(self):
        """验证错误码与 HTTP 状态码对应"""
        self.assertEqual(ErrorCode.OK, 200)
        self.assertEqual(ErrorCode.BAD_REQUEST, 400)
        self.assertEqual(ErrorCode.UNAUTHORIZED, 401)
        self.assertEqual(ErrorCode.NOT_FOUND, 404)
        self.assertEqual(ErrorCode.CONFLICT, 409)
        self.assertEqual(ErrorCode.INTERNAL_ERROR, 500)


class TestStandardResponse(unittest.TestCase):
    """测试标准响应格式"""
    
    def test_success_response(self):
        """成功响应必须包含 ok/code/message/request_id/data"""
        resp = StandardResponse.success(data={"foo": "bar"}, request_id="test123")
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["code"], ErrorCode.OK)
        self.assertEqual(resp["message"], "ok")
        self.assertEqual(resp["request_id"], "test123")
        self.assertEqual(resp["data"], {"foo": "bar"})
    
    def test_success_auto_request_id(self):
        """未提供 request_id 时自动生成"""
        resp = StandardResponse.success()
        self.assertIn("request_id", resp)
        self.assertIsInstance(resp["request_id"], str)
        self.assertGreater(len(resp["request_id"]), 0)
    
    def test_error_response(self):
        """错误响应必须包含 ok/code/message/request_id"""
        resp = StandardResponse.error(
            ErrorCode.BAD_REQUEST,
            "validation failed",
            request_id="test456"
        )
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["code"], 400)
        self.assertEqual(resp["message"], "validation failed")
        self.assertEqual(resp["request_id"], "test456")
    
    def test_error_with_details(self):
        """错误响应可包含 error_type 和 details"""
        resp = StandardResponse.error(
            ErrorCode.UNPROCESSABLE,
            "invalid field",
            error_type="ValidationError",
            details={"field": "email"}
        )
        self.assertEqual(resp["error_type"], "ValidationError")
        self.assertEqual(resp["details"], {"field": "email"})


class TestExceptionMapping(unittest.TestCase):
    """测试异常到错误码的映射"""
    
    def test_value_error_maps_to_bad_request(self):
        """ValueError 映射为 400"""
        code, msg = map_exception_to_error(ValueError("invalid param"))
        self.assertEqual(code, ErrorCode.BAD_REQUEST)
        self.assertIn("invalid param", msg)
    
    def test_key_error_maps_to_bad_request(self):
        """KeyError 映射为 400"""
        code, msg = map_exception_to_error(KeyError("missing_field"))
        self.assertEqual(code, ErrorCode.BAD_REQUEST)
        self.assertIn("missing_field", msg)
    
    def test_permission_error_maps_to_forbidden(self):
        """PermissionError 映射为 403"""
        code, msg = map_exception_to_error(PermissionError("access denied"))
        self.assertEqual(code, ErrorCode.FORBIDDEN)
        self.assertIn("access denied", msg)
    
    def test_file_not_found_maps_to_not_found(self):
        """FileNotFoundError 映射为 404"""
        code, msg = map_exception_to_error(FileNotFoundError("file.txt"))
        self.assertEqual(code, ErrorCode.NOT_FOUND)
        self.assertIn("file.txt", msg)
    
    def test_unknown_exception_maps_to_internal_error(self):
        """未分类异常映射为 500"""
        code, msg = map_exception_to_error(RuntimeError("unexpected"))
        self.assertEqual(code, ErrorCode.INTERNAL_ERROR)
        self.assertIn("RuntimeError", msg)
        self.assertIn("unexpected", msg)


class TestRequestId(unittest.TestCase):
    """测试 request_id 生成"""
    
    def test_generate_request_id_format(self):
        """request_id 应为 8 位字符串（UUID4 前缀）"""
        rid = _generate_request_id()
        self.assertIsInstance(rid, str)
        self.assertEqual(len(rid), 8)
    
    def test_generate_unique_ids(self):
        """连续生成的 request_id 应不同"""
        ids = {_generate_request_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)  # 无碰撞


if __name__ == "__main__":
    unittest.main()
