import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class SessionManager {
  static const _storage = FlutterSecureStorage();
  static const String _sessionKey = "sessionid";

  static Future<void> saveSession(String sessionId) async {
    await _storage.write(key: _sessionKey, value: sessionId);
  }

  static Future<String?> getSession() async {
    return await _storage.read(key: _sessionKey);
  }

  static Future<void> removeSession() async {
    await _storage.delete(key: _sessionKey);
  }
}
