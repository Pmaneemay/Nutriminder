import 'package:flutter/material.dart';



class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  late final TextEditingController _email;
  late final TextEditingController _password;

  @override
  void initState() {
   _email = TextEditingController();
   _password = TextEditingController();
    super.initState();
  }

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
      super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          
          TextField(
            controller: _email,
            autocorrect: false,
            enableSuggestions: false,
            decoration: InputDecoration(
              hintText:  "Enter Username"
            ),
          ),

          TextField(
            controller: _password,
            autocorrect: false,
            enableSuggestions: false,
            obscureText: true,
            decoration: InputDecoration(
              hintText: "Enter Password"
            ),
          ),
          TextButton(onPressed: (){

          }, child: const Text('Login'))

        ],
      ),
    );
  }
}